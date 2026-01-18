from collections import OrderedDict
from dataclasses import dataclass, field
import itertools
import threading
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple, Union
import warnings
import torch
from torch import Tensor, nn
from fairscale.nn.model_parallel import get_pipeline_parallel_group
from . import microbatch
from .async_pipeline import AsyncPipeline
from .async_schedule import Invocation, Location, ModuleWrapper
from .batchnorm import DeferredBatchNorm
from .skip.layout import SkipLayout
from .skip.skippable import Skippable
from .types import LazyModule
def instantiate_partition(self, module: Union[nn.Sequential, List[LazyModule]], balance: List[int], group: torch.distributed.ProcessGroup) -> List[ModuleWrapper]:
    layers: NamedModules = OrderedDict()

    def maybe_realize(layer: Any) -> nn.Module:
        if isinstance(layer, nn.Module):
            return layer
        elif callable(layer):
            return layer()
        else:
            raise TypeError(f'layer must be nn.Module or callable, is {type(layer)}')

    def iterate_module(module: Union[nn.Sequential, list]) -> Iterable[Tuple[Any, nn.Module]]:
        if isinstance(module, nn.Sequential):
            yield from module.named_children()
        else:
            yield from ((str(k), v) for k, v in enumerate(module))
    module_ids = list(map(id, module))
    index_of_first_use = [module_ids.index(x) for x in module_ids]
    locations: List[Location] = []
    module_iter = enumerate(iterate_module(module))
    partitions: List[List[PartitionInfo]] = []
    for bi, b in enumerate(balance):
        modules_for_rank: List[PartitionInfo] = []
        current_module: OrderedDict[str, nn.Module] = OrderedDict()

        def current_location() -> Location:
            return Location(bi, len(modules_for_rank))

        def append_module(mod: 'OrderedDict[str, nn.Module]') -> None:
            modules_for_rank.append(PartitionInfo(current_location(), mod))
        while sum(map(len, modules_for_rank)) + len(current_module) < b:
            module_index, (name, layer) = next(module_iter)
            if index_of_first_use[module_index] != module_index:
                locations.append(locations[index_of_first_use[module_index]])
                continue
            is_reused = index_of_first_use.count(index_of_first_use[module_index]) > 1
            if is_reused and len(current_module) > 0:
                append_module(current_module)
                current_module = OrderedDict()
            current_module[str(name)] = layer
            locations.append(current_location())
            if is_reused:
                append_module(current_module)
                current_module = OrderedDict()
        if len(current_module) > 0:
            append_module(current_module)
        partitions.append(modules_for_rank)
    filtered_locations: List[Optional[Location]] = [loc for loc, _ in itertools.groupby(locations)]
    filtered_locations.append(None)
    for i in range(len(filtered_locations) - 1):
        loc = filtered_locations[i]
        assert loc
        if i == 0:
            inv = Invocation(i, loc, None, filtered_locations[i + 1])
        else:
            inv = Invocation(i, loc, filtered_locations[i - 1], filtered_locations[i + 1])
        partitions[loc.stage][loc.index].invocations.append(inv)
    invocations = enumerate(iterate_module(module))
    partition = partitions[group.rank()]
    result: List[ModuleWrapper] = []
    for partition_info in partition:
        wrapper = ModuleWrapper(nn.Sequential(OrderedDict(((k, maybe_realize(m)) for k, m in partition_info.modules.items()))), partition_info.location, partition_info.invocations)
        if not isinstance(module, nn.Sequential):
            for layer in wrapper.module:
                if isinstance(layer, Skippable):
                    raise ValueError("Can't use Skippable layers with multi-process pipe and lazy construction")
        result.append(wrapper)
    return result
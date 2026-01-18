import warnings
from typing import Callable, cast, Optional, Sequence, Tuple
import torch
import torch.distributed._functional_collectives as funcol
import torch.distributed._tensor.dispatch as op_dispatch
import torch.distributed._tensor.random as random
import torch.nn as nn
from torch.distributed._tensor._collective_utils import mesh_broadcast
from torch.distributed._tensor._utils import compute_global_tensor_info
from torch.distributed._tensor.placement_types import (
from torch.distributed._tensor.random import (
from torch.distributed._tensor.redistribute import (
from torch.distributed.device_mesh import _mesh_resources, DeviceMesh
def distribute_module(module: nn.Module, device_mesh: Optional[DeviceMesh]=None, partition_fn: Optional[Callable[[str, nn.Module, DeviceMesh], None]]=None, input_fn: Optional[Callable[..., None]]=None, output_fn: Optional[Callable[..., None]]=None) -> nn.Module:
    """
    This function converts all module parameters to :class:`DTensor` parameters
    according to the `partition_fn` specified. It could also control the input or
    output of the module by specifying the `input_fn` and `output_fn`. (i.e. convert
    the input to :class:`DTensor`, convert the output back to torch.Tensor)
    Args:
        module (:class:`nn.Module`): user module to be partitioned.
        device_mesh (:class:`DeviceMesh`): the device mesh to place the module.
        partition_fn (Callable): the function to partition parameters (i.e. shard certain
            parameters across the `device_mesh`). If `partition_fn` is not specified,
            by default we replicate all module parameters of `module` across the mesh.
        input_fn (Callable): specify the input distribution, i.e. could control how the
            input of the module is sharded. `input_fn` will be installed as a module
            `forward_pre_hook` (pre forward hook).
        output_fn (Callable): specify the output distribution, i.e. could control how the
            output is sharded, or convert it back to torch.Tensor. output_fn will be
            installed as a module `forward_hook` (post forward hook).

    Returns:
        A module that contains parameters/buffers that are all `DTensor`s.
    """
    torch._C._log_api_usage_once('torch.dtensor.distribute_module')
    device_mesh = device_mesh or _mesh_resources.get_current_mesh()

    def replicate_module_params_buffers(m: nn.Module, mesh: DeviceMesh) -> None:
        full_replicate = [Replicate()] * mesh.ndim
        for key, param in m._parameters.items():
            if param is not None and (not isinstance(param, DTensor)):
                m.register_parameter(key, nn.Parameter(distribute_tensor(param.data, mesh, full_replicate)))
        for key, buffer in m._buffers.items():
            if buffer is not None and (not isinstance(buffer, DTensor)):
                m._buffers[key] = distribute_tensor(buffer, mesh, full_replicate)
    if partition_fn is None:
        for name, submod in module.named_modules():
            replicate_module_params_buffers(submod, device_mesh)
    else:
        for name, submod in module.named_modules():
            partition_fn(name, submod, device_mesh)
            replicate_module_params_buffers(submod, device_mesh)
    if input_fn is not None:
        module.register_forward_pre_hook(lambda _, inputs: input_fn(inputs, device_mesh))
    if output_fn is not None:
        module.register_forward_hook(lambda mod, inputs, outputs: output_fn(outputs, device_mesh))
    return module
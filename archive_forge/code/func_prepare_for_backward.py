import itertools
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sized, Union, cast
import torch
from torch import Tensor
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.utils.data import DistributedSampler, Sampler
from typing_extensions import Self, override
from lightning_fabric.utilities.distributed import _DatasetSamplerWrapper
from pytorch_lightning.utilities.rank_zero import rank_zero_debug, rank_zero_info
from pytorch_lightning.utilities.types import _SizedIterable
def prepare_for_backward(model: DistributedDataParallel, output: Any) -> None:
    if torch.is_grad_enabled() and model.require_backward_grad_sync:
        model.require_forward_param_sync = True
        args = list(_find_tensors(output)) if model.find_unused_parameters and (not model.static_graph) else []
        reducer = cast(torch._C._distributed_c10d.Reducer, model.reducer)
        reducer._rebuild_buckets()
        reducer.prepare_for_backward(args)
    else:
        model.require_forward_param_sync = False
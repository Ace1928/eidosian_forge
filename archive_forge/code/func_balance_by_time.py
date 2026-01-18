from typing import Any, List, Union, Sequence
import torch
from torch import Tensor
import torch.nn as nn
from . import blockpartition
from .profile import profile_sizes, profile_times
def balance_by_time(partitions: int, module: nn.Sequential, sample: Union[List[Any], Tensor], *, timeout: float=1.0, device: Device=torch.device('cuda')) -> List[int]:
    """Naive automatic balancing by elapsed time per layer.
    ::

        sample = torch.empty(128, 3, 224, 224)
        balance = balance_by_time(torch.cuda.device_count(), model, sample)
        pipe = Pipe(model, balance, chunks=8)

    Args:
        partitions (int):
            intended number of partitions
        module (torch.nn.Sequential):
            sequential module to be partitioned
        sample (torch.Tensor):
            example input with arbitrary batch size

    Keyword Args:
        timeout (float):
            profiling iterates again if the timeout (in second) is not exceeded
            (default: ``1.0``)
        device ('cpu' or 'cuda' device):
            CPU or CUDA device where each layer is profiled (default: the
            current CUDA device)

    Returns:
        A list of number of layers in each partition. Use it for the `balance`
        parameter of :class:`~torchpipe.Pipe`.

    .. note::
        `module` and `sample` must be placed on the same device.

    """
    times = profile_times(module, sample, timeout, torch.device(device))
    return balance_cost(times, partitions)
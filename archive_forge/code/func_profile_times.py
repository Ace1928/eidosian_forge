import copy
import time
from typing import Any, Generator, List, Union, Sequence
import torch
from torch import Tensor
import torch.nn as nn
from ..microbatch import Batch
def profile_times(module: nn.Sequential, sample: Union[List[Any], Tensor], timeout: float, device: torch.device) -> List[int]:
    """Profiles elapsed times per layer."""
    if any((p.grad is not None for p in module.parameters())):
        raise ValueError('some parameter already has gradient')
    _batch = Batch(sample)
    for i, x in enumerate(_batch):
        _batch[i] = x.detach().to(device).requires_grad_(x.requires_grad)
    time_bufs: List[List[float]] = [[] for _ in module]
    begun_at = time.time()
    while time.time() - begun_at < timeout:
        batch = _batch
        for i, layer in enumerate(layerwise_sandbox(module, device)):
            detach(batch)
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            tick = time.time()
            batch = batch.call(layer)
            backward_tensors = tuple((y for y in batch if y.requires_grad))
            if backward_tensors:
                torch.autograd.backward(backward_tensors, backward_tensors)
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            tock = time.time()
            time_bufs[i].append(tock - tick)
    us = 1000000
    return [sum((int(t * us) for t in buf)) for buf in time_bufs]
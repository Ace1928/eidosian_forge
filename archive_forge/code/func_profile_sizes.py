import copy
import time
from typing import Any, Generator, List, Union, Sequence
import torch
from torch import Tensor
import torch.nn as nn
from ..microbatch import Batch
def profile_sizes(module: nn.Sequential, input: Union[List[Any], Tensor], chunks: int, param_scale: float, device: torch.device) -> List[int]:
    """Profiles CUDA memory usage per layer."""
    if device.type != 'cuda':
        raise ValueError('size profiler supports only CUDA device')
    batch = Batch(input)
    sizes: List[int] = []
    latent_scale = batch[0].size(0) / chunks
    for i, x in enumerate(batch):
        batch[i] = x[:1].detach().to(device).requires_grad_(x.requires_grad)
    for layer in layerwise_sandbox(module, device):
        detach(batch)
        torch._C._cuda_clearCublasWorkspaces()
        memory_before = torch.cuda.memory_allocated(device)
        batch = batch.call(layer)
        torch._C._cuda_clearCublasWorkspaces()
        memory_after = torch.cuda.memory_allocated(device)
        latent_size = memory_after - memory_before
        param_size = sum((p._typed_storage()._nbytes() for p in layer.parameters()))
        size = latent_size * latent_scale + param_size * param_scale
        sizes.append(int(size))
    return sizes
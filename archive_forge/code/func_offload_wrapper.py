import warnings
from enum import auto, Enum
from functools import partial
from typing import Any, Callable, Dict, Iterator, Optional, Tuple
import torch
import torch.nn as nn
from torch.autograd.graph import save_on_cpu
from torch.distributed.utils import _pack_kwargs, _replace_by_prefix, _unpack_kwargs
from torch.utils.checkpoint import checkpoint as torch_utils_checkpoint
def offload_wrapper(module: torch.nn.Module) -> torch.nn.Module:
    """
    Wrap a module for activation offloading to CPU.

    Offloads intermediate activations to the CPU for modules wrapped with this function.
    Wrappers with activation offload can be composed with ones that do recomputation-based
    checkpoint to trade off increased compute versus increased CPU
    memory usage and additional H2D transfers.

    Usage::
        offloaded_module = offload_wrapper(module)
        outputs = checkpointed_module(inputs)
    Args:
        module (nn.Module):
            The module to be wrapped
    Returns:
        (nn.Module):
            Wrapped module
    """
    return OffloadWrapper(module)
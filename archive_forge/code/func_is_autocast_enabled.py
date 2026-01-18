from contextlib import contextmanager
from dataclasses import dataclass
import functools
import threading
from typing import Any, Dict, Generator, Optional, Tuple
import weakref
import torch
from torch import Tensor
import torch.nn as nn
import torch.utils.checkpoint as torch_checkpoint
from fairscale.internal.containers import pack_kwargs, split_non_tensors, unpack_kwargs, unpack_non_tensors
from .checkpoint_utils import patch_batchnorm
def is_autocast_enabled() -> bool:
    """Similar to torch.is_autocast_enabled, but compatible with torch 1.5.1"""
    if hasattr(torch, 'is_autocast_enabled'):
        return torch.is_autocast_enabled()
    return False
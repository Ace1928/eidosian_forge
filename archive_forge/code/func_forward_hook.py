from contextlib import contextmanager, nullcontext
from typing import Any, Tuple
import torch
import torch.nn as nn
from torch.utils.checkpoint import (
from .contract import contract
def forward_hook(module: nn.Module, inputs: Tuple[Any, ...], output: Any) -> Any:
    if checkpoint.state(module).enable_hook:
        try:
            next(checkpoint.state(module)._ac_generator)
        except StopIteration:
            pass
        else:
            raise RuntimeError('Expected non-reentrant activation checkpoint generator to be exhausted, but it was not!')
    checkpoint.state(module)._ac_generator = None
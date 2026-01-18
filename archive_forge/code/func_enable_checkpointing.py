from collections import deque
from contextlib import contextmanager
import threading
from typing import (
import torch
from torch import Tensor
import torch.autograd
from .dependency import fork, join
from .microbatch import Batch
from .phony import get_phony
@contextmanager
def enable_checkpointing() -> Generator[None, None, None]:
    """Make :func:`is_checkpointing` return :data:`True` within a context."""
    orig = thread_local.is_checkpointing
    thread_local.is_checkpointing = True
    try:
        yield
    finally:
        thread_local.is_checkpointing = orig
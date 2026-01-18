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
def is_checkpointing() -> bool:
    """Whether the current forward propagation is under checkpointing.

    Returns:
        bool: :data:`True` if it's under checkpointing.

    """
    return thread_local.is_checkpointing
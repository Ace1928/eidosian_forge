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
def is_recomputing() -> bool:
    """Whether the current forward propagation is under checkpoint recomputation.

    Use this to prevent duplicated side-effects at forward
    propagation::

        class Counter(nn.Module):
            def __init__(self):
                super().__init__()
                self.counter = 0

            def forward(self, input):
                if not is_recomputing():
                    self.counter += 1
                return input

    Returns:
        bool: :data:`True` if it's under checkpoint recomputation.

    .. seealso:: :ref:`Detecting Recomputation`

    """
    return thread_local.is_recomputing
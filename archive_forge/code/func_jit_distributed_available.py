import builtins
import functools
import inspect
from abc import ABC, abstractmethod
from contextlib import contextmanager
from copy import deepcopy
from typing import Any, Callable, ClassVar, Dict, Generator, List, Optional, Sequence, Tuple, Union
import torch
from lightning_utilities import apply_to_collection
from torch import Tensor
from torch.nn import Module
from torchmetrics.utilities.data import (
from torchmetrics.utilities.distributed import gather_all_tensors
from torchmetrics.utilities.exceptions import TorchMetricsUserError
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE, plot_single_or_multi_val
from torchmetrics.utilities.prints import rank_zero_warn
def jit_distributed_available() -> bool:
    """Determine if distributed mode is initialized."""
    return torch.distributed.is_available() and torch.distributed.is_initialized()
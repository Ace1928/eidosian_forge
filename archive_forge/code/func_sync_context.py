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
@contextmanager
def sync_context(self, dist_sync_fn: Optional[Callable]=None, process_group: Optional[Any]=None, should_sync: bool=True, should_unsync: bool=True, distributed_available: Optional[Callable]=None) -> Generator:
    """Context manager to synchronize states.

        This context manager is used in distributed setting and makes sure that the local cache states are restored
        after yielding the synchronized state.

        Args:
            dist_sync_fn: Function to be used to perform states synchronization
            process_group:
                Specify the process group on which synchronization is called.
                default: `None` (which selects the entire world)
            should_sync: Whether to apply to state synchronization. This will have an impact
                only when running in a distributed setting.
            should_unsync: Whether to restore the cache state so that the metrics can
                continue to be accumulated.
            distributed_available: Function to determine if we are running inside a distributed setting

        """
    self.sync(dist_sync_fn=dist_sync_fn, process_group=process_group, should_sync=should_sync, distributed_available=distributed_available)
    yield
    self.unsync(should_unsync=self._is_synced and should_unsync)
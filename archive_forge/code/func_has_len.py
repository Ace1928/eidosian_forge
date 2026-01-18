import functools
import inspect
import os
from collections import OrderedDict
from contextlib import contextmanager
from functools import partial
from typing import Any, Callable, Dict, Generator, Iterable, Optional, Sized, Tuple, Type, Union
from lightning_utilities.core.inheritance import get_all_subclasses
from torch.utils.data import BatchSampler, DataLoader, IterableDataset, Sampler
from typing_extensions import TypeGuard
from lightning_fabric.utilities.enums import LightningEnum
from lightning_fabric.utilities.exceptions import MisconfigurationException
from lightning_fabric.utilities.rank_zero import rank_zero_warn
from lightning_fabric.utilities.seed import pl_worker_init_function
def has_len(dataloader: object) -> TypeGuard[Sized]:
    """Checks if a given object has ``__len__`` method implemented."""
    length = sized_len(dataloader)
    if length == 0:
        rank_zero_warn(f'`{dataloader.__class__.__name__}` returned 0 length. Please make sure this was your intention.')
    if length is not None and has_iterable_dataset(dataloader):
        rank_zero_warn('Your `IterableDataset` has `__len__` defined. In combination with multi-process data loading (when num_workers > 1), `__len__` could be inaccurate if each worker is not configured independently to avoid having duplicate data.')
    return length is not None
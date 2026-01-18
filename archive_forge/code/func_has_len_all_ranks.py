import inspect
from dataclasses import fields
from typing import Any, Dict, Generator, Iterable, Mapping, Optional, Sized, Tuple, Union
import torch
from lightning_utilities.core.apply_func import is_dataclass_instance
from torch import Tensor
from torch.utils.data import BatchSampler, DataLoader, IterableDataset, RandomSampler, Sampler, SequentialSampler
from typing_extensions import TypeGuard
import pytorch_lightning as pl
from lightning_fabric.utilities.data import (
from pytorch_lightning.overrides.distributed import _IndexBatchSamplerWrapper
from pytorch_lightning.trainer.states import RunningStage
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.rank_zero import WarningCache, rank_zero_warn
def has_len_all_ranks(dataloader: object, strategy: 'pl.strategies.Strategy', allow_zero_length_dataloader_with_multiple_devices: bool=False) -> TypeGuard[Sized]:
    """Checks if a given object has ``__len__`` method implemented on all ranks."""
    local_length = sized_len(dataloader)
    if local_length is None:
        return False
    total_length = strategy.reduce(torch.tensor(local_length, device=strategy.root_device), reduce_op='sum')
    if total_length == 0:
        rank_zero_warn(f'Total length of `{type(dataloader).__name__}` across ranks is zero. Please make sure this was your intention.')
    if total_length > 0 and local_length == 0:
        dataloader_cls_name = type(dataloader).__name__
        if not allow_zero_length_dataloader_with_multiple_devices:
            raise RuntimeError(f'`{dataloader_cls_name}` within local rank has zero length. Please make sure that it returns at least 1 batch.')
        rank_zero_warn(f'Total length of `{dataloader_cls_name}` across ranks is zero, but local rank has zero length. Please be cautious of uneven batch length.')
    if has_iterable_dataset(dataloader):
        rank_zero_warn('Your `IterableDataset` has `__len__` defined. In combination with multi-process data loading (when num_workers > 1), `__len__` could be inaccurate if each worker is not configured independently to avoid having duplicate data.')
    return True
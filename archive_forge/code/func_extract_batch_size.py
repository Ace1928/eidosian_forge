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
def extract_batch_size(batch: BType) -> int:
    """Unpack a batch to find a ``torch.Tensor``.

    Returns:
        ``len(tensor)`` when found, or ``1`` when it hits an empty or non iterable.

    """
    error_msg = 'We could not infer the batch_size from the batch. Either simplify its structure or provide the batch_size as `self.log(..., batch_size=batch_size)`.'
    batch_size = None
    try:
        for bs in _extract_batch_size(batch):
            if batch_size is None:
                batch_size = bs
            elif batch_size != bs:
                warning_cache.warn(f'Trying to infer the `batch_size` from an ambiguous collection. The batch size we found is {batch_size}. To avoid any miscalculations, use `self.log(..., batch_size=batch_size)`.')
                break
    except RecursionError:
        raise RecursionError(error_msg)
    if batch_size is None:
        raise MisconfigurationException(error_msg)
    return batch_size
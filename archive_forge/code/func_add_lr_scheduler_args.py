import os
import sys
from functools import partial, update_wrapper
from types import MethodType
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, Type, Union
import torch
from lightning_utilities.core.imports import RequirementCache
from lightning_utilities.core.rank_zero import _warn
from torch.optim import Optimizer
from typing_extensions import override
import pytorch_lightning as pl
from lightning_fabric.utilities.cloud_io import get_filesystem
from lightning_fabric.utilities.types import _TORCH_LRSCHEDULER
from pytorch_lightning import Callback, LightningDataModule, LightningModule, Trainer, seed_everything
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.model_helpers import is_overridden
from pytorch_lightning.utilities.rank_zero import rank_zero_warn
def add_lr_scheduler_args(self, lr_scheduler_class: Union[LRSchedulerType, Tuple[LRSchedulerType, ...]]=LRSchedulerTypeTuple, nested_key: str='lr_scheduler', link_to: str='AUTOMATIC') -> None:
    """Adds arguments from a learning rate scheduler class to a nested key of the parser.

        Args:
            lr_scheduler_class: Any subclass of ``torch.optim.lr_scheduler.{_LRScheduler, ReduceLROnPlateau}``. Use
                tuple to allow subclasses.
            nested_key: Name of the nested namespace to store arguments.
            link_to: Dot notation of a parser key to set arguments or AUTOMATIC.

        """
    if isinstance(lr_scheduler_class, tuple):
        assert all((issubclass(o, LRSchedulerTypeTuple) for o in lr_scheduler_class))
    else:
        assert issubclass(lr_scheduler_class, LRSchedulerTypeTuple)
    kwargs: Dict[str, Any] = {'instantiate': False, 'fail_untyped': False, 'skip': {'optimizer'}}
    if isinstance(lr_scheduler_class, tuple):
        self.add_subclass_arguments(lr_scheduler_class, nested_key, **kwargs)
    else:
        self.add_class_arguments(lr_scheduler_class, nested_key, sub_configs=True, **kwargs)
    self._lr_schedulers[nested_key] = (lr_scheduler_class, link_to)
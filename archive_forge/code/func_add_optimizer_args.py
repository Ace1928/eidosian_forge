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
def add_optimizer_args(self, optimizer_class: Union[Type[Optimizer], Tuple[Type[Optimizer], ...]]=(Optimizer,), nested_key: str='optimizer', link_to: str='AUTOMATIC') -> None:
    """Adds arguments from an optimizer class to a nested key of the parser.

        Args:
            optimizer_class: Any subclass of :class:`torch.optim.Optimizer`. Use tuple to allow subclasses.
            nested_key: Name of the nested namespace to store arguments.
            link_to: Dot notation of a parser key to set arguments or AUTOMATIC.

        """
    if isinstance(optimizer_class, tuple):
        assert all((issubclass(o, Optimizer) for o in optimizer_class))
    else:
        assert issubclass(optimizer_class, Optimizer)
    kwargs: Dict[str, Any] = {'instantiate': False, 'fail_untyped': False, 'skip': {'params'}}
    if isinstance(optimizer_class, tuple):
        self.add_subclass_arguments(optimizer_class, nested_key, **kwargs)
    else:
        self.add_class_arguments(optimizer_class, nested_key, sub_configs=True, **kwargs)
    self._optimizers[nested_key] = (optimizer_class, link_to)
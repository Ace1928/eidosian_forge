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
def add_lightning_class_args(self, lightning_class: Union[Callable[..., Union[Trainer, LightningModule, LightningDataModule, Callback]], Type[Trainer], Type[LightningModule], Type[LightningDataModule], Type[Callback]], nested_key: str, subclass_mode: bool=False, required: bool=True) -> List[str]:
    """Adds arguments from a lightning class to a nested key of the parser.

        Args:
            lightning_class: A callable or any subclass of {Trainer, LightningModule, LightningDataModule, Callback}.
            nested_key: Name of the nested namespace to store arguments.
            subclass_mode: Whether allow any subclass of the given class.
            required: Whether the argument group is required.

        Returns:
            A list with the names of the class arguments added.

        """
    if callable(lightning_class) and (not isinstance(lightning_class, type)):
        lightning_class = class_from_function(lightning_class)
    if isinstance(lightning_class, type) and issubclass(lightning_class, (Trainer, LightningModule, LightningDataModule, Callback)):
        if issubclass(lightning_class, Callback):
            self.callback_keys.append(nested_key)
        if subclass_mode:
            return self.add_subclass_arguments(lightning_class, nested_key, fail_untyped=False, required=required)
        return self.add_class_arguments(lightning_class, nested_key, fail_untyped=False, instantiate=not issubclass(lightning_class, Trainer), sub_configs=True)
    raise MisconfigurationException(f'Cannot add arguments from: {lightning_class}. You should provide either a callable or a subclass of: Trainer, LightningModule, LightningDataModule, or Callback.')
import os
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Tuple, Union
import torch.multiprocessing as mp
from torch.utils.data import BatchSampler, DataLoader, RandomSampler, Sampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
import pytorch_lightning as pl
from lightning_fabric.utilities.data import (
from lightning_fabric.utilities.distributed import DistributedSamplerWrapper
from pytorch_lightning.overrides.distributed import UnrepeatedDistributedSamplerWrapper
from pytorch_lightning.trainer import call
from pytorch_lightning.trainer.states import RunningStage, TrainerFn
from pytorch_lightning.utilities.combined_loader import CombinedLoader
from pytorch_lightning.utilities.data import _is_dataloader_shuffled, _update_dataloader
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.imports import _graphcore_available_and_importable
from pytorch_lightning.utilities.model_helpers import is_overridden
from pytorch_lightning.utilities.rank_zero import WarningCache, rank_zero_warn
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from pytorch_lightning.utilities.warnings import PossibleUserWarning
def dataloader(self) -> Union[TRAIN_DATALOADERS, EVAL_DATALOADERS]:
    """Returns the dataloader from the source.

        If the source is a module, the method with the corresponding :attr:`name` gets called.

        """
    if isinstance(self.instance, pl.LightningModule):
        return call._call_lightning_module_hook(self.instance.trainer, self.name, pl_module=self.instance)
    if isinstance(self.instance, pl.LightningDataModule):
        assert self.instance.trainer is not None
        return call._call_lightning_datamodule_hook(self.instance.trainer, self.name)
    assert self.instance is not None
    return self.instance
import math
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Dict, Generator, Optional, Union, cast
from lightning_utilities.core.imports import RequirementCache
from typing_extensions import override
import pytorch_lightning as pl
from pytorch_lightning.callbacks.progress.progress_bar import ProgressBar
from pytorch_lightning.utilities.types import STEP_OUTPUT
@override
def on_validation_end(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
    if trainer.state.fn == 'fit':
        self._update_metrics(trainer, pl_module)
    self.reset_dataloader_idx_tracker()
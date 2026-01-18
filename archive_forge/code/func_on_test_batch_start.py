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
def on_test_batch_start(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule', batch: Any, batch_idx: int, dataloader_idx: int=0) -> None:
    if self.is_disabled or not self.has_dataloader_changed(dataloader_idx):
        return
    if self.test_progress_bar_id is not None:
        assert self.progress is not None
        self.progress.update(self.test_progress_bar_id, advance=0, visible=False)
    self.test_progress_bar_id = self._add_task(self.total_test_batches_current_dataloader, self.test_description)
    self.refresh()
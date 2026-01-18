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
def on_sanity_check_end(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
    if self.progress is not None:
        assert self.val_sanity_progress_bar_id is not None
        self.progress.update(self.val_sanity_progress_bar_id, advance=0, visible=False)
    self.refresh()
import importlib
import math
import os
import sys
from typing import Any, Dict, Optional, Union
from typing_extensions import override
from pytorch_lightning.utilities.types import STEP_OUTPUT
import pytorch_lightning as pl
from pytorch_lightning.callbacks.progress.progress_bar import ProgressBar
from pytorch_lightning.utilities.rank_zero import rank_zero_debug
def init_validation_tqdm(self) -> Tqdm:
    """Override this to customize the tqdm bar for validation."""
    has_main_bar = self.trainer.state.fn != 'validate'
    return Tqdm(desc=self.validation_description, position=2 * self.process_position + has_main_bar, disable=self.is_disabled, leave=not has_main_bar, dynamic_ncols=True, file=sys.stdout, bar_format=self.BAR_FORMAT)
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
@predict_progress_bar.setter
def predict_progress_bar(self, bar: _tqdm) -> None:
    self._predict_progress_bar = bar
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
def convert_inf(x: Optional[Union[int, float]]) -> Optional[Union[int, float]]:
    """The tqdm doesn't support inf/nan values.

    We have to convert it to None.

    """
    if x is None or math.isinf(x) or math.isnan(x):
        return None
    return x
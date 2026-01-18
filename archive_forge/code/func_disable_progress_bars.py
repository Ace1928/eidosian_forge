import io
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional, Union
from tqdm.auto import tqdm as old_tqdm
from ..constants import HF_HUB_DISABLE_PROGRESS_BARS
def disable_progress_bars() -> None:
    """
    Disable globally progress bars used in `huggingface_hub` except if `HF_HUB_DISABLE_PROGRESS_BARS` environment
    variable has been set.

    Use [`~utils.enable_progress_bars`] to re-enable them.
    """
    if HF_HUB_DISABLE_PROGRESS_BARS is False:
        warnings.warn('Cannot disable progress bars: environment variable `HF_HUB_DISABLE_PROGRESS_BARS=0` is set and has priority.')
        return
    global _hf_hub_progress_bars_disabled
    _hf_hub_progress_bars_disabled = True
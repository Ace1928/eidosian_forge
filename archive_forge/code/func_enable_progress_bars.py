import io
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional, Union
from tqdm.auto import tqdm as old_tqdm
from ..constants import HF_HUB_DISABLE_PROGRESS_BARS
def enable_progress_bars() -> None:
    """
    Enable globally progress bars used in `huggingface_hub` except if `HF_HUB_DISABLE_PROGRESS_BARS` environment
    variable has been set.

    Use [`~utils.disable_progress_bars`] to disable them.
    """
    if HF_HUB_DISABLE_PROGRESS_BARS is True:
        warnings.warn('Cannot enable progress bars: environment variable `HF_HUB_DISABLE_PROGRESS_BARS=1` is set and has priority.')
        return
    global _hf_hub_progress_bars_disabled
    _hf_hub_progress_bars_disabled = False
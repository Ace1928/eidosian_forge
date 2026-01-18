import io
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional, Union
from tqdm.auto import tqdm as old_tqdm
from ..constants import HF_HUB_DISABLE_PROGRESS_BARS
def are_progress_bars_disabled() -> bool:
    """Return whether progress bars are globally disabled or not.

    Progress bars used in `huggingface_hub` can be enable or disabled globally using [`~utils.enable_progress_bars`]
    and [`~utils.disable_progress_bars`] or by setting `HF_HUB_DISABLE_PROGRESS_BARS` as environment variable.
    """
    global _hf_hub_progress_bars_disabled
    return _hf_hub_progress_bars_disabled
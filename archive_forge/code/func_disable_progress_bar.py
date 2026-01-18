import functools
import logging
import os
import sys
import threading
from logging import (
from logging import captureWarnings as _captureWarnings
from typing import Optional
import huggingface_hub.utils as hf_hub_utils
from tqdm import auto as tqdm_lib
def disable_progress_bar():
    """Disable tqdm progress bar."""
    global _tqdm_active
    _tqdm_active = False
    hf_hub_utils.disable_progress_bars()
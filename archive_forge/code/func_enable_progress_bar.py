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
def enable_progress_bar():
    """Enable tqdm progress bar."""
    global _tqdm_active
    _tqdm_active = True
    hf_hub_utils.enable_progress_bars()
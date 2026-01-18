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
def get_log_levels_dict():
    return log_levels
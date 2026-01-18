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
def enable_default_handler() -> None:
    """Enable the default handler of the HuggingFace Transformers's root logger."""
    _configure_library_root_logger()
    assert _default_handler is not None
    _get_library_root_logger().addHandler(_default_handler)
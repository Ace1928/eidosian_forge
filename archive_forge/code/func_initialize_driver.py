import abc
import hashlib
import os
import tempfile
from pathlib import Path
from ..common.build import _build
from .cache import get_cache_manager
def initialize_driver():
    import torch
    if torch.version.hip is not None:
        return HIPDriver()
    elif torch.cuda.is_available():
        return CudaDriver()
    else:
        return UnsupportedDriver()
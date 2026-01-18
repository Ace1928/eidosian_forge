import os
import time
import socket
import pathlib
import tempfile
import contextlib
from typing import Union, Optional
from functools import lru_cache
@lru_cache()
def get_torch_device_name(mps_enabled: bool=False):
    with contextlib.suppress(ImportError):
        import torch
        if torch.cuda.is_available():
            return 'cuda'
        if mps_enabled and torch.torch.backends.mps.is_available():
            return 'mps'
    return 'cpu'
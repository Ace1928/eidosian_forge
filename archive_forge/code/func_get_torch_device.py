import os
import time
import socket
import pathlib
import tempfile
import contextlib
from typing import Union, Optional
from functools import lru_cache
@lru_cache()
def get_torch_device(mps_enabled: bool=False):
    with contextlib.suppress(ImportError):
        import torch
        return torch.device(get_torch_device_name(mps_enabled))
    return 'cpu'
from __future__ import annotations
import logging
import math
import os
import threading
import warnings
from contextlib import contextmanager
from functools import partial
from typing import Any, Callable, Optional
import torch
from .utils import (
from .utils.dataclasses import SageMakerDistributedType
@property
def default_device(self) -> torch.device:
    """
        Returns the default device which is:
        - MPS if `torch.backends.mps.is_available()` and `torch.backends.mps.is_built()` both return True.
        - CUDA if `torch.cuda.is_available()`
        - NPU if `is_npu_available()`
        - CPU otherwise
        """
    if is_mps_available():
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    elif is_xpu_available():
        return torch.device('xpu:0')
    elif is_npu_available():
        return torch.device('npu')
    else:
        return torch.device('cpu')
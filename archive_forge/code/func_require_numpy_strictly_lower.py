import importlib.util
import inspect
import sys
from collections import OrderedDict
from contextlib import contextmanager
from typing import Tuple, Union
import numpy as np
from packaging import version
from transformers.utils import is_torch_available
@contextmanager
def require_numpy_strictly_lower(package_version: str, message: str):
    if not version.parse(np.__version__) < version.parse(package_version):
        raise ImportError(f'Found an incompatible version of numpy. Found version {np.__version__}, but expected numpy<{version}. {message}')
    try:
        yield
    finally:
        pass
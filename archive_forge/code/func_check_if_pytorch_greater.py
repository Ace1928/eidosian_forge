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
def check_if_pytorch_greater(target_version: str, message: str):
    """
    A context manager that does nothing except checking if the PyTorch version is greater than `pt_version`
    """
    import torch
    if not version.parse(torch.__version__) >= version.parse(target_version):
        raise ImportError(f'Found an incompatible version of PyTorch. Found version {torch.__version__}, but only {target_version} and above are supported. {message}')
    try:
        yield
    finally:
        pass
import importlib.util
import itertools
import os
import subprocess
import sys
import unittest
from collections.abc import MutableMapping
from typing import Any, Callable, Dict, Iterable, Optional, Tuple
import torch
from . import (
def is_ort_training_available():
    is_ort_train_available = importlib.util.find_spec('onnxruntime.training') is not None
    if importlib.util.find_spec('torch_ort') is not None:
        try:
            is_torch_ort_configured = True
            subprocess.run([sys.executable, '-m', 'torch_ort.configure'], shell=False, check=True)
        except subprocess.CalledProcessError:
            is_torch_ort_configured = False
    return is_ort_train_available and is_torch_ort_configured
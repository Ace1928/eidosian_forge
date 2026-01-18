import functools
import importlib
import logging
import os
import tempfile
import torch
from .common import device_from_inputs, fake_tensor_unsupported
from .registry import register_backend
def has_tvm():
    try:
        importlib.import_module('tvm')
        return True
    except ImportError:
        return False
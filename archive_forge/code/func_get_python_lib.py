import logging
import os
import torch
from . import _cpp_lib
from .checkpoint import (  # noqa: E402, F401
@compute_once
def get_python_lib():
    return torch.library.Library('xformers_python', 'DEF')
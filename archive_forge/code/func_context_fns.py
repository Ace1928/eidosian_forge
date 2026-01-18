from contextlib import contextmanager, nullcontext
from typing import Any, Tuple
import torch
import torch.nn as nn
from torch.utils.checkpoint import (
from .contract import contract
def context_fns():
    return (nullcontext(), _no_hook(module))
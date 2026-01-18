import importlib
import warnings
from typing import Any, Optional
import torch
import torch.nn as nn
import torch.nn.init as init
from peft.tuners.tuners_utils import BaseTunerLayer
from .layer import LoraLayer
@property
def is_paralle_a(self):
    warnings.warn('`is_paralle_a` is going to be deprecated in a future release. Please use `is_parallel_a`', FutureWarning)
    return self.is_parallel_a
import re
import warnings
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from functools import partial
from typing import Any, List, Tuple, Optional, Dict
import torch
import torch.nn as nn
from torch.ao.quantization.utils import (
@torch.jit.export
def reset_min_max_vals(self):
    """Resets the min/max values."""
    self.min_val = torch.rand(0)
    self.max_val = torch.rand(0)
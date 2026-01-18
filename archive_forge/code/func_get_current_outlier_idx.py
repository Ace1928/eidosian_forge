from dataclasses import dataclass
from functools import reduce  # Required in Python 3
import operator
from typing import Callable, Optional, Tuple
import warnings
from warnings import warn
import torch
import bitsandbytes.functional as F
def get_current_outlier_idx(self):
    return torch.Tensor(list(self.outliers)).to(torch.int64)
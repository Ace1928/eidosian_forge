from dataclasses import dataclass
from functools import reduce  # Required in Python 3
import operator
from typing import Callable, Optional, Tuple
import warnings
from warnings import warn
import torch
import bitsandbytes.functional as F
def reset_grads(self):
    self.CB = None
    self.CxB = None
    self.SB = None
    self.SCB = None
    self.CxBt = None
    self.SBt = None
    self.CBt = None
from __future__ import annotations
import operator
import warnings
import weakref
from contextlib import nullcontext
from enum import Enum
from functools import cmp_to_key, reduce
from typing import (
import torch
from torch import sym_float, sym_int, sym_max
def check_pin_memory(pin_memory: bool):
    torch._check_not_implemented(not pin_memory, lambda: 'PrimTorch does not support pinned memory')
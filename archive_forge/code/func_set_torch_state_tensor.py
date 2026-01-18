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
@staticmethod
def set_torch_state_tensor(seed, offset):
    seed_portion = seed.reshape([1]).view(torch.uint8)
    offset_portion = offset.reshape([1]).view(torch.uint8)
    new_state = torch.cat([seed_portion, offset_portion])
    torch.cuda.set_rng_state(new_state)
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
def apply_perm(inp, perm):
    ndim = len(inp)
    permuted_inp = [-1] * ndim
    for idx, x in enumerate(perm):
        permuted_inp[idx] = inp[x]
    return permuted_inp
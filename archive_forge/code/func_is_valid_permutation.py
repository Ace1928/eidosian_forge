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
def is_valid_permutation(rank: int, perm: DimsSequenceType) -> bool:
    """
    Validates that perm is a permutation of length rank.
    """
    if not isinstance(perm, Sequence):
        return False
    if not tuple(sorted(perm)) == tuple(range(0, rank)):
        return False
    return True
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
def is_expandable_to(shape: ShapeType, desired: ShapeType) -> bool:
    """Checks if a shape can be expanded to another shape.
    This is equivalent to checking if the two shapes are broadcastable.
    """
    if len(shape) > len(desired):
        return False
    for i in range(len(shape)):
        if shape[-i - 1] != desired[-i - 1] and shape[-i - 1] != 1:
            return False
    return True
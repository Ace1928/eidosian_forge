import dataclasses
import itertools
import sympy
from sympy.logic.boolalg import BooleanAtom, Boolean as SympyBoolean
import operator
import math
import logging
import torch
from typing import Union, Dict, Optional, SupportsFloat
from torch._prims_common import dtype_to_type
from .interp import sympy_interp
@staticmethod
def expr_cond_pair(a, b):
    assert b.is_bool, f"expect cond_expr's ValueRange to be a boolean range but got {b}"
    return (a, b)
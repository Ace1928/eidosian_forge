from __future__ import annotations
import warnings
import pytest
from datashader.macros import expand_varargs
import inspect
from numba import jit
def function_with_vararg_call_numba(a, b, *others):
    return a + b - function_no_vararg_numba(*others)
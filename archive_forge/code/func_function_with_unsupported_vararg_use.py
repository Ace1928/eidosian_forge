from __future__ import annotations
import warnings
import pytest
from datashader.macros import expand_varargs
import inspect
from numba import jit
def function_with_unsupported_vararg_use(a, b, *others):
    print(others[0])
    function_with_vararg(a, b, *others)
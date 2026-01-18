import copy
import ctypes
import importlib.util
import json
import os
import re
import sys
import warnings
import weakref
from abc import ABC, abstractmethod
from collections.abc import Mapping
from enum import IntEnum, unique
from functools import wraps
from inspect import Parameter, signature
from typing import (
import numpy as np
import scipy.sparse
from ._typing import (
from .compat import PANDAS_INSTALLED, DataFrame, py_str
from .libpath import find_lib_path
def ctypes2numpy(cptr: CNumericPtr, length: int, dtype: Type[np.number]) -> np.ndarray:
    """Convert a ctypes pointer array to a numpy array."""
    ctype: Type[CNumeric] = _numpy2ctypes_type(dtype)
    if not isinstance(cptr, ctypes.POINTER(ctype)):
        raise RuntimeError(f'expected {ctype} pointer')
    res = np.zeros(length, dtype=dtype)
    if not ctypes.memmove(res.ctypes.data, cptr, length * res.strides[0]):
        raise RuntimeError('memmove failed')
    return res
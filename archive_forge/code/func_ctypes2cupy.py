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
def ctypes2cupy(cptr: CNumericPtr, length: int, dtype: Type[np.number]) -> CupyT:
    """Convert a ctypes pointer array to a cupy array."""
    import cupy
    from cupy.cuda.memory import MemoryPointer, UnownedMemory
    CUPY_TO_CTYPES_MAPPING: Dict[Type[np.number], Type[CNumeric]] = {cupy.float32: ctypes.c_float, cupy.uint32: ctypes.c_uint}
    if dtype not in CUPY_TO_CTYPES_MAPPING:
        raise RuntimeError(f'Supported types: {CUPY_TO_CTYPES_MAPPING.keys()}')
    addr = ctypes.cast(cptr, ctypes.c_void_p).value
    device = cupy.cuda.runtime.pointerGetAttributes(addr).device
    unownd = UnownedMemory(addr, length * ctypes.sizeof(CUPY_TO_CTYPES_MAPPING[dtype]), owner=None)
    memptr = MemoryPointer(unownd, 0)
    mem = cupy.ndarray((length,), dtype=dtype, memptr=memptr)
    assert mem.device.id == device
    arr = cupy.array(mem, copy=True)
    return arr
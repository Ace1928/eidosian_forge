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
def from_pystr_to_cstr(data: Union[str, List[str]]) -> Union[bytes, ctypes.Array]:
    """Convert a Python str or list of Python str to C pointer

    Parameters
    ----------
    data
        str or list of str
    """
    if isinstance(data, str):
        return bytes(data, 'utf-8')
    if isinstance(data, list):
        data_as_bytes: List[bytes] = [bytes(d, 'utf-8') for d in data]
        pointers: ctypes.Array[ctypes.c_char_p] = (ctypes.c_char_p * len(data_as_bytes))(*data_as_bytes)
        return pointers
    raise TypeError()
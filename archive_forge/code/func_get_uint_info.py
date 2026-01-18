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
def get_uint_info(self, field: str) -> np.ndarray:
    """Get unsigned integer property from the DMatrix.

        Parameters
        ----------
        field: str
            The field name of the information

        Returns
        -------
        info : array
            a numpy array of unsigned integer information of the data
        """
    length = c_bst_ulong()
    ret = ctypes.POINTER(ctypes.c_uint)()
    _check_call(_LIB.XGDMatrixGetUIntInfo(self.handle, c_str(field), ctypes.byref(length), ctypes.byref(ret)))
    return ctypes2numpy(ret, length.value, np.uint32)
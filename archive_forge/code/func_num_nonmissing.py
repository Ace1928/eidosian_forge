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
def num_nonmissing(self) -> int:
    """Get the number of non-missing values in the DMatrix.

        .. versionadded:: 1.7.0

        """
    ret = c_bst_ulong()
    _check_call(_LIB.XGDMatrixNumNonMissing(self.handle, ctypes.byref(ret)))
    return ret.value
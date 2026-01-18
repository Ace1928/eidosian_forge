import abc
import ctypes
import inspect
import json
import warnings
from collections import OrderedDict
from copy import deepcopy
from enum import Enum
from functools import wraps
from os import SEEK_END, environ
from os.path import getsize
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, Union
import numpy as np
import scipy.sparse
from .compat import (PANDAS_INSTALLED, PYARROW_INSTALLED, arrow_cffi, arrow_is_floating, arrow_is_integer, concat,
from .libpath import find_lib_path
def model_from_string(self, model_str: str) -> 'Booster':
    """Load Booster from a string.

        Parameters
        ----------
        model_str : str
            Model will be loaded from this string.

        Returns
        -------
        self : Booster
            Loaded Booster object.
        """
    _safe_call(_LIB.LGBM_BoosterFree(self._handle))
    self._free_buffer()
    self._handle = ctypes.c_void_p()
    out_num_iterations = ctypes.c_int(0)
    _safe_call(_LIB.LGBM_BoosterLoadModelFromString(_c_str(model_str), ctypes.byref(out_num_iterations), ctypes.byref(self._handle)))
    out_num_class = ctypes.c_int(0)
    _safe_call(_LIB.LGBM_BoosterGetNumClasses(self._handle, ctypes.byref(out_num_class)))
    self.__num_class = out_num_class.value
    self.pandas_categorical = _load_pandas_categorical(model_str=model_str)
    return self
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
def model_to_string(self, num_iteration: Optional[int]=None, start_iteration: int=0, importance_type: str='split') -> str:
    """Save Booster to string.

        Parameters
        ----------
        num_iteration : int or None, optional (default=None)
            Index of the iteration that should be saved.
            If None, if the best iteration exists, it is saved; otherwise, all iterations are saved.
            If <= 0, all iterations are saved.
        start_iteration : int, optional (default=0)
            Start index of the iteration that should be saved.
        importance_type : str, optional (default="split")
            What type of feature importance should be saved.
            If "split", result contains numbers of times the feature is used in a model.
            If "gain", result contains total gains of splits which use the feature.

        Returns
        -------
        str_repr : str
            String representation of Booster.
        """
    if num_iteration is None:
        num_iteration = self.best_iteration
    importance_type_int = _FEATURE_IMPORTANCE_TYPE_MAPPER[importance_type]
    buffer_len = 1 << 20
    tmp_out_len = ctypes.c_int64(0)
    string_buffer = ctypes.create_string_buffer(buffer_len)
    ptr_string_buffer = ctypes.c_char_p(ctypes.addressof(string_buffer))
    _safe_call(_LIB.LGBM_BoosterSaveModelToString(self._handle, ctypes.c_int(start_iteration), ctypes.c_int(num_iteration), ctypes.c_int(importance_type_int), ctypes.c_int64(buffer_len), ctypes.byref(tmp_out_len), ptr_string_buffer))
    actual_len = tmp_out_len.value
    if actual_len > buffer_len:
        string_buffer = ctypes.create_string_buffer(actual_len)
        ptr_string_buffer = ctypes.c_char_p(ctypes.addressof(string_buffer))
        _safe_call(_LIB.LGBM_BoosterSaveModelToString(self._handle, ctypes.c_int(start_iteration), ctypes.c_int(num_iteration), ctypes.c_int(importance_type_int), ctypes.c_int64(actual_len), ctypes.byref(tmp_out_len), ptr_string_buffer))
    ret = string_buffer.value.decode('utf-8')
    ret += _dump_pandas_categorical(self.pandas_categorical)
    return ret
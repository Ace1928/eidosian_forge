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
@classmethod
def from_booster(cls, booster: 'Booster', pred_parameter: Dict[str, Any]) -> '_InnerPredictor':
    """Initialize an ``_InnerPredictor`` from a ``Booster``.

        Parameters
        ----------
        booster : Booster
            Booster.
        pred_parameter : dict
            Other parameters for the prediction.
        """
    out_cur_iter = ctypes.c_int(0)
    _safe_call(_LIB.LGBM_BoosterGetCurrentIteration(booster._handle, ctypes.byref(out_cur_iter)))
    return cls(booster_handle=booster._handle, pandas_categorical=booster.pandas_categorical, pred_parameter=pred_parameter, manage_handle=False)
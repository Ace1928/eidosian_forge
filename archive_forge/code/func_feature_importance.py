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
def feature_importance(self, importance_type: str='split', iteration: Optional[int]=None) -> np.ndarray:
    """Get feature importances.

        Parameters
        ----------
        importance_type : str, optional (default="split")
            How the importance is calculated.
            If "split", result contains numbers of times the feature is used in a model.
            If "gain", result contains total gains of splits which use the feature.
        iteration : int or None, optional (default=None)
            Limit number of iterations in the feature importance calculation.
            If None, if the best iteration exists, it is used; otherwise, all trees are used.
            If <= 0, all trees are used (no limits).

        Returns
        -------
        result : numpy array
            Array with feature importances.
        """
    if iteration is None:
        iteration = self.best_iteration
    importance_type_int = _FEATURE_IMPORTANCE_TYPE_MAPPER[importance_type]
    result = np.empty(self.num_feature(), dtype=np.float64)
    _safe_call(_LIB.LGBM_BoosterFeatureImportance(self._handle, ctypes.c_int(iteration), ctypes.c_int(importance_type_int), result.ctypes.data_as(ctypes.POINTER(ctypes.c_double))))
    if importance_type_int == _C_API_FEATURE_IMPORTANCE_SPLIT:
        return result.astype(np.int32)
    else:
        return result
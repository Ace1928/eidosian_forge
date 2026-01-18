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
def feature_num_bin(self, feature: Union[int, str]) -> int:
    """Get the number of bins for a feature.

        .. versionadded:: 4.0.0

        Parameters
        ----------
        feature : int or str
            Index or name of the feature.

        Returns
        -------
        number_of_bins : int
            The number of constructed bins for the feature in the Dataset.
        """
    if self._handle is not None:
        if isinstance(feature, str):
            feature_index = self.feature_name.index(feature)
        else:
            feature_index = feature
        ret = ctypes.c_int(0)
        _safe_call(_LIB.LGBM_DatasetGetFeatureNumBin(self._handle, ctypes.c_int(feature_index), ctypes.byref(ret)))
        return ret.value
    else:
        raise LightGBMError('Cannot get feature_num_bin before construct dataset')
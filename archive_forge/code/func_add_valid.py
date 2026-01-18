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
def add_valid(self, data: Dataset, name: str) -> 'Booster':
    """Add validation data.

        Parameters
        ----------
        data : Dataset
            Validation data.
        name : str
            Name of validation data.

        Returns
        -------
        self : Booster
            Booster with set validation data.
        """
    if not isinstance(data, Dataset):
        raise TypeError(f'Validation data should be Dataset instance, met {type(data).__name__}')
    if data._predictor is not self.__init_predictor:
        raise LightGBMError('Add validation data failed, you should use same predictor for these data')
    _safe_call(_LIB.LGBM_BoosterAddValidData(self._handle, data.construct()._handle))
    self.valid_sets.append(data)
    self.name_valid_sets.append(name)
    self.__num_dataset += 1
    self.__inner_predict_buffer.append(None)
    self.__is_predicted_cur_iter.append(False)
    return self
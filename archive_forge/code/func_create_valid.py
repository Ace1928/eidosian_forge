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
def create_valid(self, data: _LGBM_TrainDataType, label: Optional[_LGBM_LabelType]=None, weight: Optional[_LGBM_WeightType]=None, group: Optional[_LGBM_GroupType]=None, init_score: Optional[_LGBM_InitScoreType]=None, params: Optional[Dict[str, Any]]=None, position: Optional[_LGBM_PositionType]=None) -> 'Dataset':
    """Create validation data align with current Dataset.

        Parameters
        ----------
        data : str, pathlib.Path, numpy array, pandas DataFrame, H2O DataTable's Frame, scipy.sparse, Sequence, list of Sequence or list of numpy array
            Data source of Dataset.
            If str or pathlib.Path, it represents the path to a text file (CSV, TSV, or LibSVM) or a LightGBM Dataset binary file.
        label : list, numpy 1-D array, pandas Series / one-column DataFrame, pyarrow Array, pyarrow ChunkedArray or None, optional (default=None)
            Label of the data.
        weight : list, numpy 1-D array, pandas Series, pyarrow Array, pyarrow ChunkedArray or None, optional (default=None)
            Weight for each instance. Weights should be non-negative.
        group : list, numpy 1-D array, pandas Series, pyarrow Array, pyarrow ChunkedArray or None, optional (default=None)
            Group/query data.
            Only used in the learning-to-rank task.
            sum(group) = n_samples.
            For example, if you have a 100-document dataset with ``group = [10, 20, 40, 10, 10, 10]``, that means that you have 6 groups,
            where the first 10 records are in the first group, records 11-30 are in the second group, records 31-70 are in the third group, etc.
        init_score : list, list of lists (for multi-class task), numpy array, pandas Series, pandas DataFrame (for multi-class task), pyarrow Array, pyarrow ChunkedArray, pyarrow Table (for multi-class task) or None, optional (default=None)
            Init score for Dataset.
        params : dict or None, optional (default=None)
            Other parameters for validation Dataset.
        position : numpy 1-D array, pandas Series or None, optional (default=None)
            Position of items used in unbiased learning-to-rank task.

        Returns
        -------
        valid : Dataset
            Validation Dataset with reference to self.
        """
    ret = Dataset(data, label=label, reference=self, weight=weight, group=group, position=position, init_score=init_score, params=params, free_raw_data=self.free_raw_data)
    ret._predictor = self._predictor
    ret.pandas_categorical = self.pandas_categorical
    return ret
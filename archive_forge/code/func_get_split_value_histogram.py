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
def get_split_value_histogram(self, feature: Union[int, str], bins: Optional[Union[int, str]]=None, xgboost_style: bool=False) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray, pd_DataFrame]:
    """Get split value histogram for the specified feature.

        Parameters
        ----------
        feature : int or str
            The feature name or index the histogram is calculated for.
            If int, interpreted as index.
            If str, interpreted as name.

            .. warning::

                Categorical features are not supported.

        bins : int, str or None, optional (default=None)
            The maximum number of bins.
            If None, or int and > number of unique split values and ``xgboost_style=True``,
            the number of bins equals number of unique split values.
            If str, it should be one from the list of the supported values by ``numpy.histogram()`` function.
        xgboost_style : bool, optional (default=False)
            Whether the returned result should be in the same form as it is in XGBoost.
            If False, the returned value is tuple of 2 numpy arrays as it is in ``numpy.histogram()`` function.
            If True, the returned value is matrix, in which the first column is the right edges of non-empty bins
            and the second one is the histogram values.

        Returns
        -------
        result_tuple : tuple of 2 numpy arrays
            If ``xgboost_style=False``, the values of the histogram of used splitting values for the specified feature
            and the bin edges.
        result_array_like : numpy array or pandas DataFrame (if pandas is installed)
            If ``xgboost_style=True``, the histogram of used splitting values for the specified feature.
        """

    def add(root: Dict[str, Any]) -> None:
        """Recursively add thresholds."""
        if 'split_index' in root:
            if feature_names is not None and isinstance(feature, str):
                split_feature = feature_names[root['split_feature']]
            else:
                split_feature = root['split_feature']
            if split_feature == feature:
                if isinstance(root['threshold'], str):
                    raise LightGBMError('Cannot compute split value histogram for the categorical feature')
                else:
                    values.append(root['threshold'])
            add(root['left_child'])
            add(root['right_child'])
    model = self.dump_model()
    feature_names = model.get('feature_names')
    tree_infos = model['tree_info']
    values: List[float] = []
    for tree_info in tree_infos:
        add(tree_info['tree_structure'])
    if bins is None or (isinstance(bins, int) and xgboost_style):
        n_unique = len(np.unique(values))
        bins = max(min(n_unique, bins) if bins is not None else n_unique, 1)
    hist, bin_edges = np.histogram(values, bins=bins)
    if xgboost_style:
        ret = np.column_stack((bin_edges[1:], hist))
        ret = ret[ret[:, 1] > 0]
        if PANDAS_INSTALLED:
            return pd_DataFrame(ret, columns=['SplitValue', 'Count'])
        else:
            return ret
    else:
        return (hist, bin_edges)
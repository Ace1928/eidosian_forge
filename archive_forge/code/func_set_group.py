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
def set_group(self, group: Optional[_LGBM_GroupType]) -> 'Dataset':
    """Set group size of Dataset (used for ranking).

        Parameters
        ----------
        group : list, numpy 1-D array, pandas Series, pyarrow Array, pyarrow ChunkedArray or None
            Group/query data.
            Only used in the learning-to-rank task.
            sum(group) = n_samples.
            For example, if you have a 100-document dataset with ``group = [10, 20, 40, 10, 10, 10]``, that means that you have 6 groups,
            where the first 10 records are in the first group, records 11-30 are in the second group, records 31-70 are in the third group, etc.

        Returns
        -------
        self : Dataset
            Dataset with set group.
        """
    self.group = group
    if self._handle is not None and group is not None:
        if not _is_pyarrow_array(group):
            group = _list_to_1d_numpy(group, dtype=np.int32, name='group')
        self.set_field('group', group)
        constructed_group = self.get_field('group')
        if constructed_group is not None:
            self.group = np.diff(constructed_group)
    return self
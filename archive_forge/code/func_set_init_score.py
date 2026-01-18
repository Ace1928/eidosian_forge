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
def set_init_score(self, init_score: Optional[_LGBM_InitScoreType]) -> 'Dataset':
    """Set init score of Booster to start from.

        Parameters
        ----------
        init_score : list, list of lists (for multi-class task), numpy array, pandas Series, pandas DataFrame (for multi-class task), pyarrow Array, pyarrow ChunkedArray, pyarrow Table (for multi-class task) or None
            Init score for Booster.

        Returns
        -------
        self : Dataset
            Dataset with set init score.
        """
    self.init_score = init_score
    if self._handle is not None and init_score is not None:
        self.set_field('init_score', init_score)
        self.init_score = self.get_field('init_score')
    return self
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
def set_leaf_output(self, tree_id: int, leaf_id: int, value: float) -> 'Booster':
    """Set the output of a leaf.

        .. versionadded:: 4.0.0

        Parameters
        ----------
        tree_id : int
            The index of the tree.
        leaf_id : int
            The index of the leaf in the tree.
        value : float
            Value to set as the output of the leaf.

        Returns
        -------
        self : Booster
            Booster with the leaf output set.
        """
    _safe_call(_LIB.LGBM_BoosterSetLeafValue(self._handle, ctypes.c_int(tree_id), ctypes.c_int(leaf_id), ctypes.c_double(value)))
    return self
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
def free_dataset(self) -> 'Booster':
    """Free Booster's Datasets.

        Returns
        -------
        self : Booster
            Booster without Datasets.
        """
    self.__dict__.pop('train_set', None)
    self.__dict__.pop('valid_sets', None)
    self.__num_dataset = 0
    return self
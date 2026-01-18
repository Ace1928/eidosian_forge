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
def register_logger(logger: Any, info_method_name: str='info', warning_method_name: str='warning') -> None:
    """Register custom logger.

    Parameters
    ----------
    logger : Any
        Custom logger.
    info_method_name : str, optional (default="info")
        Method used to log info messages.
    warning_method_name : str, optional (default="warning")
        Method used to log warning messages.
    """
    if not _has_method(logger, info_method_name) or not _has_method(logger, warning_method_name):
        raise TypeError(f"Logger must provide '{info_method_name}' and '{warning_method_name}' method")
    global _LOGGER, _INFO_METHOD_NAME, _WARNING_METHOD_NAME
    _LOGGER = logger
    _INFO_METHOD_NAME = info_method_name
    _WARNING_METHOD_NAME = warning_method_name
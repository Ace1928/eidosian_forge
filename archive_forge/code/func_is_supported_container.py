import importlib
from functools import wraps
from typing import Protocol, runtime_checkable
import numpy as np
from scipy.sparse import issparse
from .._config import get_config
from ._available_if import available_if
def is_supported_container(self, X):
    pl = check_library_installed('polars')
    return isinstance(X, pl.DataFrame)
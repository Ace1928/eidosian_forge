from __future__ import annotations
import types
from typing import (
import numpy as np
from pandas.compat._optional import import_optional_dependency
from pandas.errors import NumbaUtilError
def set_use_numba(enable: bool=False) -> None:
    global GLOBAL_USE_NUMBA
    if enable:
        import_optional_dependency('numba')
    GLOBAL_USE_NUMBA = enable
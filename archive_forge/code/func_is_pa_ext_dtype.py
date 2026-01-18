import ctypes
import json
import os
import warnings
from typing import Any, Callable, Iterator, List, Optional, Sequence, Tuple, cast
import numpy as np
from ._typing import (
from .compat import DataFrame, lazy_isinstance
from .core import (
def is_pa_ext_dtype(dtype: Any) -> bool:
    """Return whether dtype is a pyarrow extension type for pandas"""
    return hasattr(dtype, 'pyarrow_dtype')
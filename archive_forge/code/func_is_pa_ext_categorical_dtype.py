import ctypes
import json
import os
import warnings
from typing import Any, Callable, Iterator, List, Optional, Sequence, Tuple, cast
import numpy as np
from ._typing import (
from .compat import DataFrame, lazy_isinstance
from .core import (
def is_pa_ext_categorical_dtype(dtype: Any) -> bool:
    """Check whether dtype is a dictionary type."""
    return lazy_isinstance(getattr(dtype, 'pyarrow_dtype', None), 'pyarrow.lib', 'DictionaryType')
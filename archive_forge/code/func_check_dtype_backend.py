from __future__ import annotations
from collections.abc import (
from typing import (
import numpy as np
from pandas._libs import lib
from pandas.core.dtypes.common import (
def check_dtype_backend(dtype_backend) -> None:
    if dtype_backend is not lib.no_default:
        if dtype_backend not in ['numpy_nullable', 'pyarrow']:
            raise ValueError(f"dtype_backend {dtype_backend} is invalid, only 'numpy_nullable' and 'pyarrow' are allowed.")
from __future__ import annotations
import contextlib
import functools
import inspect
import io
import itertools
import math
import os
import re
import sys
import warnings
from collections.abc import (
from enum import Enum
from pathlib import Path
from typing import (
import numpy as np
import pandas as pd
from xarray.namedarray.utils import (  # noqa: F401
def contains_only_chunked_or_numpy(obj) -> bool:
    """Returns True if xarray object contains only numpy arrays or chunked arrays (i.e. pure dask or cubed).

    Expects obj to be Dataset or DataArray"""
    from xarray.core.dataarray import DataArray
    from xarray.namedarray.pycompat import is_chunked_array
    if isinstance(obj, DataArray):
        obj = obj._to_temp_dataset()
    return all([isinstance(var.data, np.ndarray) or is_chunked_array(var.data) for var in obj.variables.values()])
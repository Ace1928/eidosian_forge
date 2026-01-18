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
def maybe_coerce_to_str(index, original_coords):
    """maybe coerce a pandas Index back to a nunpy array of type str

    pd.Index uses object-dtype to store str - try to avoid this for coords
    """
    from xarray.core import dtypes
    try:
        result_type = dtypes.result_type(*original_coords)
    except TypeError:
        pass
    else:
        if result_type.kind in 'SU':
            index = np.asarray(index, dtype=result_type.type)
    return index
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
def is_valid_numpy_dtype(dtype: Any) -> bool:
    try:
        np.dtype(dtype)
    except (TypeError, ValueError):
        return False
    else:
        return True
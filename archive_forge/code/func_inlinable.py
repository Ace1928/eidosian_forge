from __future__ import annotations
import math
import numbers
from collections.abc import Iterable
from enum import Enum
from typing import Any
from dask import config, core, utils
from dask.base import normalize_token, tokenize
from dask.core import (
from dask.typing import Graph, Key
def inlinable(v):
    try:
        return functions_of(v).issubset(fast_functions)
    except TypeError:
        return False
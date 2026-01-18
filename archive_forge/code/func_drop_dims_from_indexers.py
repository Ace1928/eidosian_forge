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
def drop_dims_from_indexers(indexers: Mapping[Any, Any], dims: Iterable[Hashable] | Mapping[Any, int], missing_dims: ErrorOptionsWithWarn) -> Mapping[Hashable, Any]:
    """Depending on the setting of missing_dims, drop any dimensions from indexers that
    are not present in dims.

    Parameters
    ----------
    indexers : dict
    dims : sequence
    missing_dims : {"raise", "warn", "ignore"}
    """
    if missing_dims == 'raise':
        invalid = indexers.keys() - set(dims)
        if invalid:
            raise ValueError(f'Dimensions {invalid} do not exist. Expected one or more of {dims}')
        return indexers
    elif missing_dims == 'warn':
        indexers = dict(indexers)
        invalid = indexers.keys() - set(dims)
        if invalid:
            warnings.warn(f'Dimensions {invalid} do not exist. Expected one or more of {dims}')
        for key in invalid:
            indexers.pop(key)
        return indexers
    elif missing_dims == 'ignore':
        return {key: val for key, val in indexers.items() if key in dims}
    else:
        raise ValueError(f'Unrecognised option {missing_dims} for missing_dims argument')
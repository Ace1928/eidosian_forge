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
def parse_ordered_dims(dim: Dims, all_dims: tuple[Hashable, ...], *, check_exists: bool=True, replace_none: bool=True) -> tuple[Hashable, ...] | None | ellipsis:
    """Parse one or more dimensions.

    A single dimension must be always a str, multiple dimensions
    can be Hashables. This supports e.g. using a tuple as a dimension.
    An ellipsis ("...") in a sequence of dimensions will be
    replaced with all remaining dimensions. This only makes sense when
    the input is a sequence and not e.g. a set.

    Parameters
    ----------
    dim : str, Sequence of Hashable or "...", "..." or None
        Dimension(s) to parse. If "..." appears in a Sequence
        it always gets replaced with all remaining dims
    all_dims : tuple of Hashable
        All possible dimensions.
    check_exists: bool, default: True
        if True, check if dim is a subset of all_dims.
    replace_none : bool, default: True
        If True, return all_dims if dim is None.

    Returns
    -------
    parsed_dims : tuple of Hashable
        Input dimensions as a tuple.
    """
    if dim is not None and dim is not ... and (not isinstance(dim, str)) and (... in dim):
        dims_set: set[Hashable | ellipsis] = set(dim)
        all_dims_set = set(all_dims)
        if check_exists:
            _check_dims(dims_set, all_dims_set)
        if len(all_dims_set) != len(all_dims):
            raise ValueError('Cannot use ellipsis with repeated dims')
        dims = tuple(dim)
        if dims.count(...) > 1:
            raise ValueError('More than one ellipsis supplied')
        other_dims = tuple((d for d in all_dims if d not in dims_set))
        idx = dims.index(...)
        return dims[:idx] + other_dims + dims[idx + 1:]
    else:
        return parse_dims(dim=dim, all_dims=all_dims, check_exists=check_exists, replace_none=replace_none)
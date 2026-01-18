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
def parse_dims(dim: Dims, all_dims: tuple[Hashable, ...], *, check_exists: bool=True, replace_none: bool=True) -> tuple[Hashable, ...] | None | ellipsis:
    """Parse one or more dimensions.

    A single dimension must be always a str, multiple dimensions
    can be Hashables. This supports e.g. using a tuple as a dimension.
    If you supply e.g. a set of dimensions the order cannot be
    conserved, but for sequences it will be.

    Parameters
    ----------
    dim : str, Iterable of Hashable, "..." or None
        Dimension(s) to parse.
    all_dims : tuple of Hashable
        All possible dimensions.
    check_exists: bool, default: True
        if True, check if dim is a subset of all_dims.
    replace_none : bool, default: True
        If True, return all_dims if dim is None or "...".

    Returns
    -------
    parsed_dims : tuple of Hashable
        Input dimensions as a tuple.
    """
    if dim is None or dim is ...:
        if replace_none:
            return all_dims
        return dim
    if isinstance(dim, str):
        dim = (dim,)
    if check_exists:
        _check_dims(set(dim), set(all_dims))
    return tuple(dim)
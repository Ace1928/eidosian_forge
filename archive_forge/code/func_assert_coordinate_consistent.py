from __future__ import annotations
from collections.abc import Hashable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import (
import numpy as np
import pandas as pd
from xarray.core import formatting
from xarray.core.alignment import Aligner
from xarray.core.indexes import (
from xarray.core.merge import merge_coordinates_without_align, merge_coords
from xarray.core.types import DataVars, Self, T_DataArray, T_Xarray
from xarray.core.utils import (
from xarray.core.variable import Variable, as_variable, calculate_dimensions
def assert_coordinate_consistent(obj: T_Xarray, coords: Mapping[Any, Variable]) -> None:
    """Make sure the dimension coordinate of obj is consistent with coords.

    obj: DataArray or Dataset
    coords: Dict-like of variables
    """
    for k in obj.dims:
        if k in coords and k in obj.coords and (not coords[k].equals(obj[k].variable)):
            raise IndexError(f'dimension coordinate {k!r} conflicts between indexed and indexing objects:\n{obj[k]}\nvs.\n{coords[k]}')
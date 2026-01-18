from __future__ import annotations
import functools
import itertools
import warnings
from collections.abc import Hashable, Iterable, MutableMapping
from typing import TYPE_CHECKING, Any, Callable, Generic, Literal, TypeVar, cast
import numpy as np
from xarray.core.formatting import format_item
from xarray.core.types import HueStyleOptions, T_DataArrayOrSet
from xarray.plot.utils import (
def set_axis_labels(self, *axlabels: Hashable) -> None:
    """Set axis labels on the left column and bottom row of the grid."""
    from xarray.core.dataarray import DataArray
    for var, axis in zip(axlabels, ['x', 'y', 'z']):
        if var is not None:
            if isinstance(var, DataArray):
                getattr(self, f'set_{axis}labels')(label_from_attrs(var))
            else:
                getattr(self, f'set_{axis}labels')(str(var))
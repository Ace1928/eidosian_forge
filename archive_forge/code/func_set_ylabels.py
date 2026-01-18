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
def set_ylabels(self, label: None | str=None, **kwargs: Any) -> None:
    """Label the y axis on the left column of the grid."""
    self._set_labels('y', self._left_axes, label, **kwargs)
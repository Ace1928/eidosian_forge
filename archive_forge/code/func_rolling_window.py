from __future__ import annotations
import copy
import itertools
import math
import numbers
import warnings
from collections.abc import Hashable, Mapping, Sequence
from datetime import timedelta
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Literal, NoReturn, cast
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
import xarray as xr  # only for Dataset and DataArray
from xarray.core import common, dtypes, duck_array_ops, indexing, nputils, ops, utils
from xarray.core.arithmetic import VariableArithmetic
from xarray.core.common import AbstractArray
from xarray.core.indexing import (
from xarray.core.options import OPTIONS, _get_keep_attrs
from xarray.core.utils import (
from xarray.namedarray.core import NamedArray, _raise_if_any_duplicate_dimensions
from xarray.namedarray.pycompat import integer_types, is_0d_dask_array, to_duck_array
def rolling_window(self, dim, window, window_dim, center=False, fill_value=dtypes.NA):
    """
        Make a rolling_window along dim and add a new_dim to the last place.

        Parameters
        ----------
        dim : str
            Dimension over which to compute rolling_window.
            For nd-rolling, should be list of dimensions.
        window : int
            Window size of the rolling
            For nd-rolling, should be list of integers.
        window_dim : str
            New name of the window dimension.
            For nd-rolling, should be list of strings.
        center : bool, default: False
            If True, pad fill_value for both ends. Otherwise, pad in the head
            of the axis.
        fill_value
            value to be filled.

        Returns
        -------
        Variable that is a view of the original array with a added dimension of
        size w.
        The return dim: self.dims + (window_dim, )
        The return shape: self.shape + (window, )

        Examples
        --------
        >>> v = Variable(("a", "b"), np.arange(8).reshape((2, 4)))
        >>> v.rolling_window("b", 3, "window_dim")
        <xarray.Variable (a: 2, b: 4, window_dim: 3)> Size: 192B
        array([[[nan, nan,  0.],
                [nan,  0.,  1.],
                [ 0.,  1.,  2.],
                [ 1.,  2.,  3.]],
        <BLANKLINE>
               [[nan, nan,  4.],
                [nan,  4.,  5.],
                [ 4.,  5.,  6.],
                [ 5.,  6.,  7.]]])

        >>> v.rolling_window("b", 3, "window_dim", center=True)
        <xarray.Variable (a: 2, b: 4, window_dim: 3)> Size: 192B
        array([[[nan,  0.,  1.],
                [ 0.,  1.,  2.],
                [ 1.,  2.,  3.],
                [ 2.,  3., nan]],
        <BLANKLINE>
               [[nan,  4.,  5.],
                [ 4.,  5.,  6.],
                [ 5.,  6.,  7.],
                [ 6.,  7., nan]]])
        """
    if fill_value is dtypes.NA:
        dtype, fill_value = dtypes.maybe_promote(self.dtype)
        var = duck_array_ops.astype(self, dtype, copy=False)
    else:
        dtype = self.dtype
        var = self
    if utils.is_scalar(dim):
        for name, arg in zip(['window', 'window_dim', 'center'], [window, window_dim, center]):
            if not utils.is_scalar(arg):
                raise ValueError(f"Expected {name}={arg!r} to be a scalar like 'dim'.")
        dim = (dim,)
    nroll = len(dim)
    if utils.is_scalar(window):
        window = [window] * nroll
    if utils.is_scalar(window_dim):
        window_dim = [window_dim] * nroll
    if utils.is_scalar(center):
        center = [center] * nroll
    if len(dim) != len(window) or len(dim) != len(window_dim) or len(dim) != len(center):
        raise ValueError(f"'dim', 'window', 'window_dim', and 'center' must be the same length. Received dim={dim!r}, window={window!r}, window_dim={window_dim!r}, and center={center!r}.")
    pads = {}
    for d, win, cent in zip(dim, window, center):
        if cent:
            start = win // 2
            end = win - 1 - start
            pads[d] = (start, end)
        else:
            pads[d] = (win - 1, 0)
    padded = var.pad(pads, mode='constant', constant_values=fill_value)
    axis = self.get_axis_num(dim)
    new_dims = self.dims + tuple(window_dim)
    return Variable(new_dims, duck_array_ops.sliding_window_view(padded.data, window_shape=window, axis=axis))
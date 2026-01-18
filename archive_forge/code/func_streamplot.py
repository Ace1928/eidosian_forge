from __future__ import annotations
import functools
import inspect
import warnings
from collections.abc import Hashable, Iterable
from typing import TYPE_CHECKING, Any, Callable, TypeVar, overload
from xarray.core.alignment import broadcast
from xarray.plot import dataarray_plot
from xarray.plot.facetgrid import _easy_facetgrid
from xarray.plot.utils import (
@_dsplot
def streamplot(ds: Dataset, x: Hashable, y: Hashable, ax: Axes, u: Hashable, v: Hashable, **kwargs: Any) -> LineCollection:
    """Plot streamlines of Dataset variables.

    Wraps :py:func:`matplotlib:matplotlib.pyplot.streamplot`.
    """
    import matplotlib as mpl
    if x is None or y is None or u is None or (v is None):
        raise ValueError('Must specify x, y, u, v for streamplot plots.')
    xdim = ds[x].dims[0] if len(ds[x].dims) == 1 else None
    ydim = ds[y].dims[0] if len(ds[y].dims) == 1 else None
    if xdim is not None and ydim is None:
        ydims = set(ds[y].dims) - {xdim}
        if len(ydims) == 1:
            ydim = next(iter(ydims))
    if ydim is not None and xdim is None:
        xdims = set(ds[x].dims) - {ydim}
        if len(xdims) == 1:
            xdim = next(iter(xdims))
    dx, dy, du, dv = broadcast(ds[x], ds[y], ds[u], ds[v])
    if xdim is not None and ydim is not None:
        dx = dx.transpose(ydim, xdim)
        dy = dy.transpose(ydim, xdim)
        du = du.transpose(ydim, xdim)
        dv = dv.transpose(ydim, xdim)
    hue = kwargs.pop('hue')
    cmap_params = kwargs.pop('cmap_params')
    if hue:
        kwargs['color'] = ds[hue].values
        if not cmap_params['norm']:
            cmap_params['norm'] = mpl.colors.Normalize(cmap_params.pop('vmin'), cmap_params.pop('vmax'))
    kwargs.pop('hue_style')
    hdl = ax.streamplot(dx.values, dy.values, du.values, dv.values, **kwargs, **cmap_params)
    return hdl.lines
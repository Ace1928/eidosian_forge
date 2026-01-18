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
def map_dataarray(self: T_FacetGrid, func: Callable, x: Hashable | None, y: Hashable | None, **kwargs: Any) -> T_FacetGrid:
    """
        Apply a plotting function to a 2d facet's subset of the data.

        This is more convenient and less general than ``FacetGrid.map``

        Parameters
        ----------
        func : callable
            A plotting function with the same signature as a 2d xarray
            plotting method such as `xarray.plot.imshow`
        x, y : string
            Names of the coordinates to plot on x, y axes
        **kwargs
            additional keyword arguments to func

        Returns
        -------
        self : FacetGrid object

        """
    if kwargs.get('cbar_ax', None) is not None:
        raise ValueError('cbar_ax not supported by FacetGrid.')
    cmap_params, cbar_kwargs = _process_cmap_cbar_kwargs(func, self.data.to_numpy(), **kwargs)
    self._cmap_extend = cmap_params.get('extend')
    func_kwargs = {k: v for k, v in kwargs.items() if k not in {'cmap', 'colors', 'cbar_kwargs', 'levels'}}
    func_kwargs.update(cmap_params)
    func_kwargs['add_colorbar'] = False
    if func.__name__ != 'surface':
        func_kwargs['add_labels'] = False
    x, y = _infer_xy_labels(darray=self.data.loc[self.name_dicts.flat[0]], x=x, y=y, imshow=func.__name__ == 'imshow', rgb=kwargs.get('rgb', None))
    for d, ax in zip(self.name_dicts.flat, self.axs.flat):
        if d is not None:
            subset = self.data.loc[d]
            mappable = func(subset, x=x, y=y, ax=ax, **func_kwargs, _is_facetgrid=True)
            self._mappables.append(mappable)
    self._finalize_grid(x, y)
    if kwargs.get('add_colorbar', True):
        self.add_colorbar(**cbar_kwargs)
    return self
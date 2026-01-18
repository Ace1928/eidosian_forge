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
def map_plot1d(self: T_FacetGrid, func: Callable, x: Hashable | None, y: Hashable | None, *, z: Hashable | None=None, hue: Hashable | None=None, markersize: Hashable | None=None, linewidth: Hashable | None=None, **kwargs: Any) -> T_FacetGrid:
    """
        Apply a plotting function to a 1d facet's subset of the data.

        This is more convenient and less general than ``FacetGrid.map``

        Parameters
        ----------
        func :
            A plotting function with the same signature as a 1d xarray
            plotting method such as `xarray.plot.scatter`
        x, y :
            Names of the coordinates to plot on x, y axes
        **kwargs
            additional keyword arguments to func

        Returns
        -------
        self : FacetGrid object

        """
    self.data = self.data.copy()
    if kwargs.get('cbar_ax', None) is not None:
        raise ValueError('cbar_ax not supported by FacetGrid.')
    if func.__name__ == 'scatter':
        size_ = kwargs.pop('_size', markersize)
        size_r = _MARKERSIZE_RANGE
    else:
        size_ = kwargs.pop('_size', linewidth)
        size_r = _LINEWIDTH_RANGE
    coords_to_plot: MutableMapping[str, Hashable | None] = dict(x=x, z=z, hue=hue, size=size_)
    coords_to_plot = _guess_coords_to_plot(self.data, coords_to_plot, kwargs)
    hue = coords_to_plot['hue']
    hueplt = self.data.coords[hue] if hue else None
    hueplt_norm = _Normalize(hueplt)
    self._hue_var = hueplt
    cbar_kwargs = kwargs.pop('cbar_kwargs', {})
    if hueplt_norm.data is not None:
        if not hueplt_norm.data_is_numeric:
            cbar_kwargs.update(format=hueplt_norm.format, ticks=hueplt_norm.ticks)
            kwargs.update(levels=hueplt_norm.levels)
        cmap_params, cbar_kwargs = _process_cmap_cbar_kwargs(func, cast('DataArray', hueplt_norm.values).data, cbar_kwargs=cbar_kwargs, **kwargs)
        self._cmap_extend = cmap_params.get('extend')
    else:
        cmap_params = {}
    size_ = coords_to_plot['size']
    sizeplt = self.data.coords[size_] if size_ else None
    sizeplt_norm = _Normalize(data=sizeplt, width=size_r)
    if sizeplt_norm.data is not None:
        self.data[size_] = sizeplt_norm.values
    func_kwargs = {k: v for k, v in kwargs.items() if k not in {'cmap', 'colors', 'cbar_kwargs', 'levels'}}
    func_kwargs.update(cmap_params)
    func_kwargs['add_colorbar'] = False
    func_kwargs['add_legend'] = False
    func_kwargs['add_title'] = False
    add_labels_ = np.zeros(self.axs.shape + (3,), dtype=bool)
    if kwargs.get('z') is not None:
        add_labels_[:] = True
    else:
        add_labels_[-1, :, 0] = True
        add_labels_[:, 0, 1] = True
    if self._single_group:
        full = tuple(({self._single_group: x} for x in range(0, self.data[self._single_group].size)))
        empty = tuple((None for x in range(self._nrow * self._ncol - len(full))))
        name_d = full + empty
    else:
        rowcols = itertools.product(range(0, self.data[self._row_var].size), range(0, self.data[self._col_var].size))
        name_d = tuple(({self._row_var: r, self._col_var: c} for r, c in rowcols))
    name_dicts = np.array(name_d).reshape(self._nrow, self._ncol)
    for add_lbls, d, ax in zip(add_labels_.reshape((self.axs.size, -1)), name_dicts.flat, self.axs.flat):
        func_kwargs['add_labels'] = add_lbls
        if d is not None:
            subset = self.data.isel(d)
            mappable = func(subset, x=x, y=y, ax=ax, hue=hue, _size=size_, **func_kwargs, _is_facetgrid=True)
            self._mappables.append(mappable)
    self._finalize_grid()
    self._set_lims()
    add_colorbar, add_legend = _determine_guide(hueplt_norm, sizeplt_norm, kwargs.get('add_colorbar', None), kwargs.get('add_legend', None))
    if add_legend:
        use_legend_elements = False if func.__name__ == 'hist' else True
        if use_legend_elements:
            self.add_legend(use_legend_elements=use_legend_elements, hueplt_norm=hueplt_norm if not add_colorbar else _Normalize(None), sizeplt_norm=sizeplt_norm, primitive=self._mappables, legend_ax=self.fig, plotfunc=func.__name__)
        else:
            self.add_legend(use_legend_elements=use_legend_elements)
    if add_colorbar:
        if 'label' not in cbar_kwargs:
            cbar_kwargs['label'] = label_from_attrs(hueplt_norm.data)
        self.add_colorbar(**cbar_kwargs)
    return self
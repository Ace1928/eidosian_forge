import sys
from collections.abc import Hashable
from functools import wraps
from packaging.version import Version
from types import FunctionType
import bokeh
import numpy as np
import pandas as pd
import param
import holoviews as hv
def process_xarray(data, x, y, by, groupby, use_dask, persist, gridded, label, value_label, other_dims, kind=None):
    import xarray as xr
    if isinstance(data, xr.Dataset):
        dataset = data
    else:
        name = data.name or label or value_label
        dataset = data.to_dataset(name=name)
    all_vars = list(other_dims) if other_dims else []
    for var in [x, y, by, groupby]:
        if isinstance(var, list):
            all_vars.extend(var)
        elif isinstance(var, str):
            all_vars.append(var)
    if not gridded:
        not_found = [var for var in all_vars if var not in list(dataset.data_vars) + list(dataset.coords)]
        _, extra_vars, extra_coords = process_derived_datetime_xarray(dataset, not_found)
        dataset = dataset.assign_coords(**{var: dataset[var] for var in extra_coords})
        dataset = dataset.assign(**{var: dataset[var] for var in extra_vars})
    data_vars = list(dataset.data_vars)
    ignore = (by or []) + (groupby or [])
    dims = [c for c in dataset.coords if dataset[c].shape != () and c not in ignore][::-1]
    index_dims = [d for d in dims if d in dataset.indexes]
    if gridded:
        data = dataset
        if len(dims) < 2:
            dims += [dim for dim in list(data.dims)[::-1] if dim not in dims]
        if not (x or y):
            for c in dataset.coords:
                axis = dataset[c].attrs.get('axis', '')
                if axis.lower() == 'x':
                    x = c
                elif axis.lower() == 'y':
                    y = c
        if not (x or y):
            x, y = index_dims[:2] if len(index_dims) > 1 else dims[:2]
        elif x and (not y):
            y = [d for d in dims if d != x][0]
        elif y and (not x):
            x = [d for d in dims if d != y][0]
        if len(dims) > 2 and kind not in ('table', 'dataset') and (not groupby):
            dims = list(data.coords[x].dims) + list(data.coords[y].dims)
            groupby = [d for d in index_dims if d not in (x, y) and d not in dims and (d not in other_dims)]
    else:
        if use_dask:
            data = dataset.to_dask_dataframe()
            data = data.persist() if persist else data
        else:
            data = dataset.to_dataframe()
            if len(data.index.names) > 1:
                data = data.reset_index()
        if len(dims) == 0:
            dims = ['index']
        if x and (not y):
            y = dims[0] if x in data_vars else data_vars
        elif y and (not x):
            x = data_vars[0] if y in dims else dims[0]
        elif not x and (not y):
            x, y = (dims[0], data_vars)
        for var in [x, y]:
            if isinstance(var, list):
                all_vars.extend(var)
            elif isinstance(var, str):
                all_vars.append(var)
        covered_dims = []
        for var in all_vars:
            if var in dataset.coords:
                covered_dims.extend(dataset[var].dims)
        leftover_dims = [dim for dim in index_dims if dim not in covered_dims + all_vars]
        if groupby is None:
            groupby = [c for c in leftover_dims if c not in (by or [])]
    return (data, x, y, by, groupby)
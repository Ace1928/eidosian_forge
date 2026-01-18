from __future__ import annotations
import collections
import itertools
import operator
from collections.abc import Hashable, Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Callable, Literal, TypedDict
import numpy as np
from xarray.core.alignment import align
from xarray.core.coordinates import Coordinates
from xarray.core.dataarray import DataArray
from xarray.core.dataset import Dataset
from xarray.core.indexes import Index
from xarray.core.merge import merge
from xarray.core.utils import is_dask_collection
from xarray.core.variable import Variable
def subset_dataset_to_block(graph: dict, gname: str, dataset: Dataset, input_chunk_bounds, chunk_index):
    """
    Creates a task that subsets an xarray dataset to a block determined by chunk_index.
    Block extents are determined by input_chunk_bounds.
    Also subtasks that subset the constituent variables of a dataset.
    """
    import dask
    data_vars = []
    coords = []
    chunk_tuple = tuple(chunk_index.values())
    chunk_dims_set = set(chunk_index)
    variable: Variable
    for name, variable in dataset.variables.items():
        if dask.is_dask_collection(variable.data):
            chunk = (variable.data.name, *tuple((chunk_index[dim] for dim in variable.dims)))
            chunk_variable_task = (f'{name}-{gname}-{chunk[0]!r}',) + chunk_tuple
            graph[chunk_variable_task] = (tuple, [variable.dims, chunk, variable.attrs])
        else:
            assert name in dataset.dims or variable.ndim == 0
            subsetter = {dim: _get_chunk_slicer(dim, chunk_index, input_chunk_bounds) for dim in variable.dims}
            if set(variable.dims) < chunk_dims_set:
                this_var_chunk_tuple = tuple((chunk_index[dim] for dim in variable.dims))
            else:
                this_var_chunk_tuple = chunk_tuple
            chunk_variable_task = (f'{name}-{gname}-{dask.base.tokenize(subsetter)}',) + this_var_chunk_tuple
            if variable.ndim == 0 or chunk_variable_task not in graph:
                subset = variable.isel(subsetter)
                graph[chunk_variable_task] = (tuple, [subset.dims, subset._data, subset.attrs])
        if name in dataset._coord_names:
            coords.append([name, chunk_variable_task])
        else:
            data_vars.append([name, chunk_variable_task])
    return (Dataset, (dict, data_vars), (dict, coords), dataset.attrs)
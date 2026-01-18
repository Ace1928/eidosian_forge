import datetime
import functools
import importlib
import re
import warnings
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union
import numpy as np
import tree
import xarray as xr
from .. import __version__, utils
from ..rcparams import rcParams
def generate_dims_coords(shape, var_name, dims=None, coords=None, default_dims=None, index_origin=None, skip_event_dims=None):
    """Generate default dimensions and coordinates for a variable.

    Parameters
    ----------
    shape : tuple[int]
        Shape of the variable
    var_name : str
        Name of the variable. If no dimension name(s) is provided, ArviZ
        will generate a default dimension name using ``var_name``, e.g.,
        ``"foo_dim_0"`` for the first dimension if ``var_name`` is ``"foo"``.
    dims : list
        List of dimensions for the variable
    coords : dict[str] -> list[str]
        Map of dimensions to coordinates
    default_dims : list[str]
        Dimension names that are not part of the variable's shape. For example,
        when manipulating Monte Carlo traces, the ``default_dims`` would be
        ``["chain" , "draw"]`` which ArviZ uses as its own names for dimensions
        of MCMC traces.
    index_origin : int, optional
        Starting value of integer coordinate values. Defaults to the value in rcParam
        ``data.index_origin``.
    skip_event_dims : bool, default False

    Returns
    -------
    list[str]
        Default dims
    dict[str] -> list[str]
        Default coords
    """
    if index_origin is None:
        index_origin = rcParams['data.index_origin']
    if default_dims is None:
        default_dims = []
    if dims is None:
        dims = []
    if skip_event_dims is None:
        skip_event_dims = False
    if coords is None:
        coords = {}
    coords = deepcopy(coords)
    dims = deepcopy(dims)
    ndims = len([dim for dim in dims if dim not in default_dims])
    if ndims > len(shape):
        if skip_event_dims:
            dims = dims[:len(shape)]
        else:
            warnings.warn(('In variable {var_name}, there are ' + 'more dims ({dims_len}) given than exist ({shape_len}). ' + 'Passed array should have shape ({defaults}*shape)').format(var_name=var_name, dims_len=len(dims), shape_len=len(shape), defaults=','.join(default_dims) + ', ' if default_dims is not None else ''), UserWarning)
    if skip_event_dims:
        for i, (dim, dim_size) in enumerate(zip(dims, shape)):
            if dim in coords and dim_size != len(coords[dim]):
                dims = dims[:i]
                break
    for i, dim_len in enumerate(shape):
        idx = i + len([dim for dim in default_dims if dim in dims])
        if len(dims) < idx + 1:
            dim_name = f'{var_name}_dim_{idx}'
            dims.append(dim_name)
        elif dims[idx] is None:
            dim_name = f'{var_name}_dim_{idx}'
            dims[idx] = dim_name
        dim_name = dims[idx]
        if dim_name not in coords:
            coords[dim_name] = np.arange(index_origin, dim_len + index_origin)
    coords = {key: coord for key, coord in coords.items() if any((key == dim for dim in dims + default_dims))}
    return (dims, coords)
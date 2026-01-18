from __future__ import annotations
import os
from collections.abc import Hashable, Iterable, Mapping, MutableMapping, Sequence
from functools import partial
from io import BytesIO
from numbers import Number
from typing import (
import numpy as np
from xarray import backends, conventions
from xarray.backends import plugins
from xarray.backends.common import (
from xarray.backends.locks import _get_scheduler
from xarray.backends.zarr import open_zarr
from xarray.core import indexing
from xarray.core.combine import (
from xarray.core.dataarray import DataArray
from xarray.core.dataset import Dataset, _get_chunk, _maybe_chunk
from xarray.core.indexes import Index
from xarray.core.types import ZarrWriteModes
from xarray.core.utils import is_remote_uri
from xarray.namedarray.daskmanager import DaskManager
from xarray.namedarray.parallelcompat import guess_chunkmanager
def open_datatree(filename_or_obj: str | os.PathLike[Any] | BufferedIOBase | AbstractDataStore, engine: T_Engine=None, **kwargs) -> DataTree:
    """
    Open and decode a DataTree from a file or file-like object, creating one tree node for each group in the file.

    Parameters
    ----------
    filename_or_obj : str, Path, file-like, or DataStore
        Strings and Path objects are interpreted as a path to a netCDF file or Zarr store.
    engine : str, optional
        Xarray backend engine to use. Valid options include `{"netcdf4", "h5netcdf", "zarr"}`.
    **kwargs : dict
        Additional keyword arguments passed to :py:func:`~xarray.open_dataset` for each group.
    Returns
    -------
    xarray.DataTree
    """
    if engine is None:
        engine = plugins.guess_engine(filename_or_obj)
    backend = plugins.get_backend(engine)
    return backend.open_datatree(filename_or_obj, **kwargs)
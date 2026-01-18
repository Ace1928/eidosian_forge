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
def dump_to_store(dataset, store, writer=None, encoder=None, encoding=None, unlimited_dims=None):
    """Store dataset contents to a backends.*DataStore object."""
    if writer is None:
        writer = ArrayWriter()
    if encoding is None:
        encoding = {}
    variables, attrs = conventions.encode_dataset_coordinates(dataset)
    check_encoding = set()
    for k, enc in encoding.items():
        variables[k].encoding = enc
        check_encoding.add(k)
    if encoder:
        variables, attrs = encoder(variables, attrs)
    store.store(variables, attrs, check_encoding, writer, unlimited_dims=unlimited_dims)
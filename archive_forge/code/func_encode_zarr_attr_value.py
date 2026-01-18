from __future__ import annotations
import json
import os
import warnings
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any
import numpy as np
from xarray import coding, conventions
from xarray.backends.common import (
from xarray.backends.store import StoreBackendEntrypoint
from xarray.core import indexing
from xarray.core.types import ZarrWriteModes
from xarray.core.utils import (
from xarray.core.variable import Variable
from xarray.namedarray.parallelcompat import guess_chunkmanager
from xarray.namedarray.pycompat import integer_types
def encode_zarr_attr_value(value):
    """
    Encode a attribute value as something that can be serialized as json

    Many xarray datasets / variables have numpy arrays and values. This
    function handles encoding / decoding of such items.

    ndarray -> list
    scalar array -> scalar
    other -> other (no change)
    """
    if isinstance(value, np.ndarray):
        encoded = value.tolist()
    elif isinstance(value, np.generic):
        encoded = value.item()
    else:
        encoded = value
    return encoded
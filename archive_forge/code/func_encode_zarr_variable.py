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
def encode_zarr_variable(var, needs_copy=True, name=None):
    """
    Converts an Variable into an Variable which follows some
    of the CF conventions:

        - Nans are masked using _FillValue (or the deprecated missing_value)
        - Rescaling via: scale_factor and add_offset
        - datetimes are converted to the CF 'units since time' format
        - dtype encodings are enforced.

    Parameters
    ----------
    var : Variable
        A variable holding un-encoded data.

    Returns
    -------
    out : Variable
        A variable which has been encoded as described above.
    """
    var = conventions.encode_cf_variable(var, name=name)
    coder = coding.strings.EncodedStringCoder(allows_unicode=True)
    var = coder.encode(var, name=name)
    var = coding.strings.ensure_fixed_length_bytes(var)
    return var
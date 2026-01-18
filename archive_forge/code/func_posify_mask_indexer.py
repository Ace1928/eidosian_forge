from __future__ import annotations
import enum
import functools
import operator
from collections import Counter, defaultdict
from collections.abc import Hashable, Iterable, Mapping
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import timedelta
from html import escape
from typing import TYPE_CHECKING, Any, Callable
import numpy as np
import pandas as pd
from xarray.core import duck_array_ops
from xarray.core.nputils import NumpyVIndexAdapter
from xarray.core.options import OPTIONS
from xarray.core.types import T_Xarray
from xarray.core.utils import (
from xarray.namedarray.parallelcompat import get_chunked_array_type
from xarray.namedarray.pycompat import array_type, integer_types, is_chunked_array
def posify_mask_indexer(indexer: ExplicitIndexer) -> ExplicitIndexer:
    """Convert masked values (-1) in an indexer to nearest unmasked values.

    This routine is useful for dask, where it can be much faster to index
    adjacent points than arbitrary points from the end of an array.

    Parameters
    ----------
    indexer : ExplicitIndexer
        Input indexer.

    Returns
    -------
    ExplicitIndexer
        Same type of input, with all values in ndarray keys equal to -1
        replaced by an adjacent non-masked element.
    """
    key = tuple((_posify_mask_subindexer(k.ravel()).reshape(k.shape) if isinstance(k, np.ndarray) else k for k in indexer.tuple))
    return type(indexer)(key)
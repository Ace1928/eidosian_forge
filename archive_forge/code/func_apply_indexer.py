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
def apply_indexer(indexable, indexer: ExplicitIndexer):
    """Apply an indexer to an indexable object."""
    if isinstance(indexer, VectorizedIndexer):
        return indexable.vindex[indexer]
    elif isinstance(indexer, OuterIndexer):
        return indexable.oindex[indexer]
    else:
        return indexable[indexer]
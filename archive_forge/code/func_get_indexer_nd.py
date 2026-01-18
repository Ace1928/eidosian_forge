from __future__ import annotations
import collections.abc
import copy
from collections import defaultdict
from collections.abc import Hashable, Iterable, Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Generic, TypeVar, cast
import numpy as np
import pandas as pd
from xarray.core import formatting, nputils, utils
from xarray.core.indexing import (
from xarray.core.utils import (
def get_indexer_nd(index, labels, method=None, tolerance=None):
    """Wrapper around :meth:`pandas.Index.get_indexer` supporting n-dimensional
    labels
    """
    flat_labels = np.ravel(labels)
    if flat_labels.dtype == 'float16':
        flat_labels = flat_labels.astype('float64')
    flat_indexer = index.get_indexer(flat_labels, method=method, tolerance=tolerance)
    indexer = flat_indexer.reshape(labels.shape)
    return indexer
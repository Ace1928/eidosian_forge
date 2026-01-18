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
def assert_chunks_compatible(a: Dataset, b: Dataset):
    a = a.unify_chunks()
    b = b.unify_chunks()
    for dim in set(a.chunks).intersection(set(b.chunks)):
        if a.chunks[dim] != b.chunks[dim]:
            raise ValueError(f'Chunk sizes along dimension {dim!r} are not equal.')
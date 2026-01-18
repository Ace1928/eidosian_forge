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
def filter_indexes_from_coords(indexes: Mapping[Any, Index], filtered_coord_names: set) -> dict[Hashable, Index]:
    """Filter index items given a (sub)set of coordinate names.

    Drop all multi-coordinate related index items for any key missing in the set
    of coordinate names.

    """
    filtered_indexes: dict[Any, Index] = dict(indexes)
    index_coord_names: dict[Hashable, set[Hashable]] = defaultdict(set)
    for name, idx in indexes.items():
        index_coord_names[id(idx)].add(name)
    for idx_coord_names in index_coord_names.values():
        if not idx_coord_names <= filtered_coord_names:
            for k in idx_coord_names:
                del filtered_indexes[k]
    return filtered_indexes
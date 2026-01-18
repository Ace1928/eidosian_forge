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
def assert_no_index_corrupted(indexes: Indexes[Index], coord_names: set[Hashable], action: str='remove coordinate(s)') -> None:
    """Assert removing coordinates or indexes will not corrupt indexes."""
    for index, index_coords in indexes.group_by_index():
        common_names = set(index_coords) & coord_names
        if common_names and len(common_names) != len(index_coords):
            common_names_str = ', '.join((f'{k!r}' for k in common_names))
            index_names_str = ', '.join((f'{k!r}' for k in index_coords))
            raise ValueError(f'cannot {action} {common_names_str}, which would corrupt the following index built from coordinates {index_names_str}:\n{index}')
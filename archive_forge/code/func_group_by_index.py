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
def group_by_index(self) -> list[tuple[T_PandasOrXarrayIndex, dict[Hashable, Variable]]]:
    """Returns a list of unique indexes and their corresponding coordinates."""
    index_coords = []
    for i in self._id_index:
        index = self._id_index[i]
        coords = {k: self._variables[k] for k in self._id_coord_names[i]}
        index_coords.append((index, coords))
    return index_coords
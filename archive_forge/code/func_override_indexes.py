from __future__ import annotations
import functools
import operator
from collections import defaultdict
from collections.abc import Hashable, Iterable, Mapping
from contextlib import suppress
from typing import TYPE_CHECKING, Any, Callable, Final, Generic, TypeVar, cast, overload
import numpy as np
import pandas as pd
from xarray.core import dtypes
from xarray.core.indexes import (
from xarray.core.types import T_Alignable
from xarray.core.utils import is_dict_like, is_full_slice
from xarray.core.variable import Variable, as_compatible_data, calculate_dimensions
def override_indexes(self) -> None:
    objects = list(self.objects)
    for i, obj in enumerate(objects[1:]):
        new_indexes = {}
        new_variables = {}
        matching_indexes = self.objects_matching_indexes[i + 1]
        for key, aligned_idx in self.aligned_indexes.items():
            obj_idx = matching_indexes.get(key)
            if obj_idx is not None:
                for name, var in self.aligned_index_vars[key].items():
                    new_indexes[name] = aligned_idx
                    new_variables[name] = var.copy(deep=self.copy)
        objects[i + 1] = obj._overwrite_indexes(new_indexes, new_variables)
    self.results = tuple(objects)
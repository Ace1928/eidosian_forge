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
def indexes_all_equal(elements: Sequence[tuple[Index, dict[Hashable, Variable]]]) -> bool:
    """Check if indexes are all equal.

    If they are not of the same type or they do not implement this check, check
    if their coordinate variables are all equal instead.

    """

    def check_variables():
        variables = [e[1] for e in elements]
        return any((not variables[0][k].equals(other_vars[k]) for other_vars in variables[1:] for k in variables[0]))
    indexes = [e[0] for e in elements]
    same_objects = all((indexes[0] is other_idx for other_idx in indexes[1:]))
    if same_objects:
        return True
    same_type = all((type(indexes[0]) is type(other_idx) for other_idx in indexes[1:]))
    if same_type:
        try:
            not_equal = any((not indexes[0].equals(other_idx) for other_idx in indexes[1:]))
        except NotImplementedError:
            not_equal = check_variables()
    else:
        not_equal = check_variables()
    return not not_equal
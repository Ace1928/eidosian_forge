from __future__ import annotations
from collections.abc import (
import itertools
from typing import (
import warnings
import numpy as np
from pandas._config import (
from pandas._libs import (
from pandas._libs.internals import (
from pandas._libs.tslibs import Timestamp
from pandas.errors import PerformanceWarning
from pandas.util._decorators import cache_readonly
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.cast import infer_dtype_from_scalar
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.missing import (
import pandas.core.algorithms as algos
from pandas.core.arrays import (
from pandas.core.arrays._mixins import NDArrayBackedExtensionArray
from pandas.core.construction import (
from pandas.core.indexers import maybe_convert_indices
from pandas.core.indexes.api import (
from pandas.core.internals.base import (
from pandas.core.internals.blocks import (
from pandas.core.internals.ops import (
def raise_construction_error(tot_items: int, block_shape: Shape, axes: list[Index], e: ValueError | None=None):
    """raise a helpful message about our construction"""
    passed = tuple(map(int, [tot_items] + list(block_shape)))
    if len(passed) <= 2:
        passed = passed[::-1]
    implied = tuple((len(ax) for ax in axes))
    if len(implied) <= 2:
        implied = implied[::-1]
    if passed == implied and e is not None:
        raise e
    if block_shape[0] == 0:
        raise ValueError('Empty data passed with indices specified.')
    raise ValueError(f'Shape of passed values is {passed}, indices imply {implied}')
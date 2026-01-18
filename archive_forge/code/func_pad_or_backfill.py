from __future__ import annotations
from functools import wraps
import inspect
import re
from typing import (
import warnings
import weakref
import numpy as np
from pandas._config import (
from pandas._libs import (
from pandas._libs.internals import (
from pandas._libs.missing import NA
from pandas._typing import (
from pandas.errors import AbstractMethodError
from pandas.util._decorators import cache_readonly
from pandas.util._exceptions import find_stack_level
from pandas.util._validators import validate_bool_kwarg
from pandas.core.dtypes.astype import (
from pandas.core.dtypes.cast import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.missing import (
from pandas.core import missing
import pandas.core.algorithms as algos
from pandas.core.array_algos.putmask import (
from pandas.core.array_algos.quantile import quantile_compat
from pandas.core.array_algos.replace import (
from pandas.core.array_algos.transforms import shift
from pandas.core.arrays import (
from pandas.core.base import PandasObject
import pandas.core.common as com
from pandas.core.computation import expressions
from pandas.core.construction import (
from pandas.core.indexers import check_setitem_lengths
from pandas.core.indexes.base import get_values_for_csv
@final
def pad_or_backfill(self, *, method: FillnaOptions, axis: AxisInt=0, inplace: bool=False, limit: int | None=None, limit_area: Literal['inside', 'outside'] | None=None, downcast: Literal['infer'] | None=None, using_cow: bool=False, already_warned=None) -> list[Block]:
    values = self.values
    kwargs: dict[str, Any] = {'method': method, 'limit': limit}
    if 'limit_area' in inspect.signature(values._pad_or_backfill).parameters:
        kwargs['limit_area'] = limit_area
    elif limit_area is not None:
        raise NotImplementedError(f'{type(values).__name__} does not implement limit_area (added in pandas 2.2). 3rd-party ExtnsionArray authors need to add this argument to _pad_or_backfill.')
    if values.ndim == 2 and axis == 1:
        new_values = values.T._pad_or_backfill(**kwargs).T
    else:
        new_values = values._pad_or_backfill(**kwargs)
    return [self.make_block_same_class(new_values)]
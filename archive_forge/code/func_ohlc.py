from __future__ import annotations
import copy
from textwrap import dedent
from typing import (
import warnings
import numpy as np
from pandas._libs import lib
from pandas._libs.tslibs import (
from pandas._libs.tslibs.dtypes import freq_to_period_freqstr
from pandas._typing import NDFrameT
from pandas.compat.numpy import function as nv
from pandas.errors import AbstractMethodError
from pandas.util._decorators import (
from pandas.util._exceptions import (
from pandas.core.dtypes.dtypes import ArrowDtype
from pandas.core.dtypes.generic import (
import pandas.core.algorithms as algos
from pandas.core.apply import (
from pandas.core.arrays import ArrowExtensionArray
from pandas.core.base import (
import pandas.core.common as com
from pandas.core.generic import (
from pandas.core.groupby.generic import SeriesGroupBy
from pandas.core.groupby.groupby import (
from pandas.core.groupby.grouper import Grouper
from pandas.core.groupby.ops import BinGrouper
from pandas.core.indexes.api import MultiIndex
from pandas.core.indexes.base import Index
from pandas.core.indexes.datetimes import (
from pandas.core.indexes.period import (
from pandas.core.indexes.timedeltas import (
from pandas.tseries.frequencies import (
from pandas.tseries.offsets import (
@final
@doc(GroupBy.ohlc)
def ohlc(self, *args, **kwargs):
    maybe_warn_args_and_kwargs(type(self), 'ohlc', args, kwargs)
    nv.validate_resampler_func('ohlc', args, kwargs)
    ax = self.ax
    obj = self._obj_with_exclusions
    if len(ax) == 0:
        obj = obj.copy()
        obj.index = _asfreq_compat(obj.index, self.freq)
        if obj.ndim == 1:
            obj = obj.to_frame()
            obj = obj.reindex(['open', 'high', 'low', 'close'], axis=1)
        else:
            mi = MultiIndex.from_product([obj.columns, ['open', 'high', 'low', 'close']])
            obj = obj.reindex(mi, axis=1)
        return obj
    return self._downsample('ohlc')
from __future__ import annotations
from contextlib import contextmanager
import operator
import numba
from numba import types
from numba.core import cgutils
from numba.core.datamodel import models
from numba.core.extending import (
from numba.core.imputils import impl_ret_borrowed
import numpy as np
from pandas._libs import lib
from pandas.core.indexes.base import Index
from pandas.core.indexing import _iLocIndexer
from pandas.core.internals import SingleBlockManager
from pandas.core.series import Series
def generate_series_binop(binop):

    @overload(binop)
    def series_binop(series1, value):
        if isinstance(series1, SeriesType):
            if isinstance(value, SeriesType):

                def series_binop_impl(series1, series2):
                    return Series(binop(series1.values, series2.values), series1.index, series1.name)
                return series_binop_impl
            else:

                def series_binop_impl(series1, value):
                    return Series(binop(series1.values, value), series1.index, series1.name)
                return series_binop_impl
    return series_binop
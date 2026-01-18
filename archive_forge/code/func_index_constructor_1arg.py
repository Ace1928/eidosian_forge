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
@lower_builtin(Index, types.Array)
def index_constructor_1arg(context, builder, sig, args):
    from numba.typed import Dict
    key_type = sig.return_type.dtype
    value_type = types.intp

    def index_impl(data):
        return Index(data, Dict.empty(key_type, value_type))
    return context.compile_internal(builder, index_impl, sig, args)
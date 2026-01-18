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
@lower_builtin(Index, types.Array, types.DictType)
def index_constructor_2arg_parent(context, builder, sig, args):
    data, hashmap = args
    index = cgutils.create_struct_proxy(sig.return_type)(context, builder)
    index.data = data
    index.hashmap = hashmap
    return impl_ret_borrowed(context, builder, sig.return_type, index._getvalue())
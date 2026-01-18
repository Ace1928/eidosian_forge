from __future__ import annotations
from contextlib import suppress
import copy
from datetime import (
import itertools
import os
import re
from textwrap import dedent
from typing import (
import warnings
import numpy as np
from pandas._config import (
from pandas._libs import (
from pandas._libs.lib import is_string_array
from pandas._libs.tslibs import timezones
from pandas.compat._optional import import_optional_dependency
from pandas.compat.pickle_compat import patch_pickle
from pandas.errors import (
from pandas.util._decorators import cache_readonly
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.missing import array_equivalent
from pandas import (
from pandas.core.arrays import (
import pandas.core.common as com
from pandas.core.computation.pytables import (
from pandas.core.construction import extract_array
from pandas.core.indexes.api import ensure_index
from pandas.core.internals import (
from pandas.io.common import stringify_path
from pandas.io.formats.printing import (
def select_as_multiple(self, keys, where=None, selector=None, columns=None, start=None, stop=None, iterator: bool=False, chunksize: int | None=None, auto_close: bool=False):
    """
        Retrieve pandas objects from multiple tables.

        .. warning::

           Pandas uses PyTables for reading and writing HDF5 files, which allows
           serializing object-dtype data with pickle when using the "fixed" format.
           Loading pickled data received from untrusted sources can be unsafe.

           See: https://docs.python.org/3/library/pickle.html for more.

        Parameters
        ----------
        keys : a list of the tables
        selector : the table to apply the where criteria (defaults to keys[0]
            if not supplied)
        columns : the columns I want back
        start : integer (defaults to None), row number to start selection
        stop  : integer (defaults to None), row number to stop selection
        iterator : bool, return an iterator, default False
        chunksize : nrows to include in iteration, return an iterator
        auto_close : bool, default False
            Should automatically close the store when finished.

        Raises
        ------
        raises KeyError if keys or selector is not found or keys is empty
        raises TypeError if keys is not a list or tuple
        raises ValueError if the tables are not ALL THE SAME DIMENSIONS
        """
    where = _ensure_term(where, scope_level=1)
    if isinstance(keys, (list, tuple)) and len(keys) == 1:
        keys = keys[0]
    if isinstance(keys, str):
        return self.select(key=keys, where=where, columns=columns, start=start, stop=stop, iterator=iterator, chunksize=chunksize, auto_close=auto_close)
    if not isinstance(keys, (list, tuple)):
        raise TypeError('keys must be a list/tuple')
    if not len(keys):
        raise ValueError('keys must have a non-zero length')
    if selector is None:
        selector = keys[0]
    tbls = [self.get_storer(k) for k in keys]
    s = self.get_storer(selector)
    nrows = None
    for t, k in itertools.chain([(s, selector)], zip(tbls, keys)):
        if t is None:
            raise KeyError(f'Invalid table [{k}]')
        if not t.is_table:
            raise TypeError(f'object [{t.pathname}] is not a table, and cannot be used in all select as multiple')
        if nrows is None:
            nrows = t.nrows
        elif t.nrows != nrows:
            raise ValueError('all tables must have exactly the same nrows!')
    _tbls = [x for x in tbls if isinstance(x, Table)]
    axis = {t.non_index_axes[0][0] for t in _tbls}.pop()

    def func(_start, _stop, _where):
        objs = [t.read(where=_where, columns=columns, start=_start, stop=_stop) for t in tbls]
        return concat(objs, axis=axis, verify_integrity=False)._consolidate()
    it = TableIterator(self, s, func, where=where, nrows=nrows, start=start, stop=stop, iterator=iterator, chunksize=chunksize, auto_close=auto_close)
    return it.get_result(coordinates=True)
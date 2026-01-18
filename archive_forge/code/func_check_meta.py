from __future__ import annotations
import math
import re
import sys
import textwrap
import traceback
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from numbers import Number
from typing import TypeVar, overload
import numpy as np
import pandas as pd
from pandas.api.types import is_dtype_equal
import dask
from dask.base import get_scheduler, is_dask_collection
from dask.core import get_deps
from dask.dataframe import (  # noqa: F401 register pandas extension types
from dask.dataframe._compat import PANDAS_GE_150, tm  # noqa: F401
from dask.dataframe.dispatch import (  # noqa : F401
from dask.dataframe.extensions import make_scalar
from dask.typing import NoDefault, no_default
from dask.utils import (
def check_meta(x, meta, funcname=None, numeric_equal=True):
    """Check that the dask metadata matches the result.

    If metadata matches, ``x`` is passed through unchanged. A nice error is
    raised if metadata doesn't match.

    Parameters
    ----------
    x : DataFrame, Series, or Index
    meta : DataFrame, Series, or Index
        The expected metadata that ``x`` should match
    funcname : str, optional
        The name of the function in which the metadata was specified. If
        provided, the function name will be included in the error message to be
        more helpful to users.
    numeric_equal : bool, optionl
        If True, integer and floating dtypes compare equal. This is useful due
        to panda's implicit conversion of integer to floating upon encountering
        missingness, which is hard to infer statically.
    """
    eq_types = {'i', 'f', 'u'} if numeric_equal else set()

    def equal_dtypes(a, b):
        if isinstance(a, pd.CategoricalDtype) != isinstance(b, pd.CategoricalDtype):
            return False
        if isinstance(a, str) and a == '-' or (isinstance(b, str) and b == '-'):
            return False
        if isinstance(a, pd.CategoricalDtype) and isinstance(b, pd.CategoricalDtype):
            if UNKNOWN_CATEGORIES in a.categories or UNKNOWN_CATEGORIES in b.categories:
                return True
            return a == b
        return a.kind in eq_types and b.kind in eq_types or is_dtype_equal(a, b)
    if not (is_dataframe_like(meta) or is_series_like(meta) or is_index_like(meta)) or is_dask_collection(meta):
        raise TypeError('Expected partition to be DataFrame, Series, or Index, got `%s`' % typename(type(meta)))
    if x.__class__ != meta.__class__:
        errmsg = 'Expected partition of type `{}` but got `{}`'.format(typename(type(meta)), typename(type(x)))
    elif is_dataframe_like(meta):
        dtypes = pd.concat([x.dtypes, meta.dtypes], axis=1, sort=True)
        bad_dtypes = [(repr(col), a, b) for col, a, b in dtypes.fillna('-').itertuples() if not equal_dtypes(a, b)]
        if bad_dtypes:
            errmsg = 'Partition type: `{}`\n{}'.format(typename(type(meta)), asciitable(['Column', 'Found', 'Expected'], bad_dtypes))
        else:
            check_matching_columns(meta, x)
            return x
    else:
        if equal_dtypes(x.dtype, meta.dtype):
            return x
        errmsg = 'Partition type: `{}`\n{}'.format(typename(type(meta)), asciitable(['', 'dtype'], [('Found', x.dtype), ('Expected', meta.dtype)]))
    raise ValueError('Metadata mismatch found%s.\n\n%s' % (' in `%s`' % funcname if funcname else '', errmsg))
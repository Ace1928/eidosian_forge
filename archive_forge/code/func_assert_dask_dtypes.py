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
def assert_dask_dtypes(ddf, res, numeric_equal=True):
    """Check that the dask metadata matches the result.

    If `numeric_equal`, integer and floating dtypes compare equal. This is
    useful due to the implicit conversion of integer to floating upon
    encountering missingness, which is hard to infer statically."""
    eq_type_sets = [{'O', 'S', 'U', 'a'}]
    if numeric_equal:
        eq_type_sets.append({'i', 'f', 'u'})

    def eq_dtypes(a, b):
        return any((a.kind in eq_types and b.kind in eq_types for eq_types in eq_type_sets)) or a == b
    if not is_dask_collection(res) and is_dataframe_like(res):
        for a, b in pd.concat([ddf._meta.dtypes, res.dtypes], axis=1).itertuples(index=False):
            assert eq_dtypes(a, b)
    elif not is_dask_collection(res) and (is_index_like(res) or is_series_like(res)):
        a = ddf._meta.dtype
        b = res.dtype
        assert eq_dtypes(a, b)
    elif hasattr(ddf._meta, 'dtype'):
        a = ddf._meta.dtype
        if not hasattr(res, 'dtype'):
            assert np.isscalar(res)
            b = np.dtype(type(res))
        else:
            b = res.dtype
        assert eq_dtypes(a, b)
    else:
        assert type(ddf._meta) == type(res)
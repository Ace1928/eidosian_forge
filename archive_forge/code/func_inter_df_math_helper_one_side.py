from __future__ import annotations
import datetime
import itertools
import json
import unittest.mock as mock
import matplotlib
import numpy as np
import pandas
import pandas._libs.lib as lib
import pytest
from numpy.testing import assert_array_equal
from pandas.core.indexing import IndexingError
from pandas.errors import SpecificationError
import modin.pandas as pd
from modin.config import Engine, NPartitions, StorageFormat
from modin.pandas.io import to_pandas
from modin.pandas.testing import assert_series_equal
from modin.tests.test_utils import warns_that_defaulting_to_pandas
from modin.utils import get_current_execution, try_cast_to_pandas
from .utils import (
def inter_df_math_helper_one_side(modin_series, pandas_series, op, comparator_kwargs=None, expected_exception=None):
    if comparator_kwargs is None:
        comparator_kwargs = {}
    try:
        pandas_attr = getattr(pandas_series, op)
    except Exception as err:
        with pytest.raises(type(err)):
            _ = getattr(modin_series, op)
        return
    modin_attr = getattr(modin_series, op)
    try:
        pandas_result = pandas_attr(4)
    except Exception as err:
        with pytest.raises(type(err)):
            try_cast_to_pandas(modin_attr(4))
    else:
        modin_result = modin_attr(4)
        df_equals(modin_result, pandas_result, **comparator_kwargs)
    try:
        pandas_result = pandas_attr(4.0)
    except Exception as err:
        with pytest.raises(type(err)):
            try_cast_to_pandas(modin_attr(4.0))
    else:
        modin_result = modin_attr(4.0)
        df_equals(modin_result, pandas_result, **comparator_kwargs)
    if op in ['__divmod__', 'divmod', 'rdivmod', 'floordiv', '__floordiv__', 'rfloordiv', '__rfloordiv__', 'mod', '__mod__', 'rmod', '__rmod__']:
        return
    eval_general(modin_series, pandas_series, lambda df: (pandas_attr if isinstance(df, pandas.Series) else modin_attr)(df), comparator_kwargs=comparator_kwargs, expected_exception=expected_exception)
    list_test = random_state.randint(RAND_LOW, RAND_HIGH, size=modin_series.shape[0])
    try:
        pandas_result = pandas_attr(list_test)
    except Exception as err:
        with pytest.raises(type(err)):
            try_cast_to_pandas(modin_attr(list_test))
    else:
        modin_result = modin_attr(list_test)
        df_equals(modin_result, pandas_result, **comparator_kwargs)
    series_test_modin = pd.Series(list_test, index=modin_series.index)
    series_test_pandas = pandas.Series(list_test, index=pandas_series.index)
    eval_general(series_test_modin, series_test_pandas, lambda df: (pandas_attr if isinstance(df, pandas.Series) else modin_attr)(df), comparator_kwargs=comparator_kwargs, expected_exception=expected_exception)
    new_idx = pandas.MultiIndex.from_tuples([(i // 4, i // 2, i) for i in modin_series.index])
    modin_df_multi_level = modin_series.copy()
    modin_df_multi_level.index = new_idx
    try:
        getattr(modin_df_multi_level, op)(modin_df_multi_level, level=1)
    except TypeError:
        pass
    else:
        with warns_that_defaulting_to_pandas():
            getattr(modin_df_multi_level, op)(modin_df_multi_level, level=1)
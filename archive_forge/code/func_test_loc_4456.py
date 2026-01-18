import os
import sys
import matplotlib
import numpy as np
import pandas
import pytest
from pandas._testing import ensure_clean
import modin.pandas as pd
from modin.config import MinPartitionSize, NPartitions, StorageFormat
from modin.pandas.indexing import is_range_like
from modin.pandas.testing import assert_index_equal
from modin.tests.pandas.utils import (
from modin.utils import get_current_execution
@pytest.mark.parametrize('key_getter, value_getter', [pytest.param(lambda df, axis: (slice(None), df.axes[axis][:2]) if axis else (df.axes[axis][:2], slice(None)), lambda df, axis: df.iloc[:, :1] if axis else df.iloc[:1, :], id='len(key)_>_len(value)'), pytest.param(lambda df, axis: (slice(None), df.axes[axis][:2]) if axis else (df.axes[axis][:2], slice(None)), lambda df, axis: df.iloc[:, :3] if axis else df.iloc[:3, :], id='len(key)_<_len(value)'), pytest.param(lambda df, axis: (slice(None), df.axes[axis][:2]) if axis else (df.axes[axis][:2], slice(None)), lambda df, axis: df.iloc[:, :2] if axis else df.iloc[:2, :], id='len(key)_==_len(value)')])
@pytest.mark.parametrize('key_axis', [0, 1])
@pytest.mark.parametrize('reverse_value_index', [True, False])
@pytest.mark.parametrize('reverse_value_columns', [True, False])
def test_loc_4456(key_getter, value_getter, key_axis, reverse_value_index, reverse_value_columns):
    data = test_data['float_nan_data']
    modin_df, pandas_df = (pd.DataFrame(data), pandas.DataFrame(data))
    key = key_getter(pandas_df, key_axis)
    if is_range_like(key[0]):
        key = (list(key[0]), key[1])
    if is_range_like(key[1]):
        key = (key[0], list(key[1]))
    value = pandas.DataFrame(np.random.randint(0, 100, size=pandas_df.shape), index=pandas_df.index, columns=pandas_df.columns)
    pdf_value = value_getter(value, key_axis)
    mdf_value = value_getter(pd.DataFrame(value), key_axis)
    if reverse_value_index:
        pdf_value = pdf_value.reindex(index=pdf_value.index[::-1])
        mdf_value = mdf_value.reindex(index=mdf_value.index[::-1])
    if reverse_value_columns:
        pdf_value = pdf_value.reindex(columns=pdf_value.columns[::-1])
        mdf_value = mdf_value.reindex(columns=mdf_value.columns[::-1])
    eval_loc(modin_df, pandas_df, pdf_value, key)
    eval_loc(modin_df, pandas_df, (mdf_value, pdf_value), key)
from __future__ import annotations
import contextlib
import datetime
import numpy as np
import pandas as pd
import pytest
import dask.dataframe as dd
from dask.dataframe._compat import PANDAS_GE_200, PANDAS_GE_210
from dask.dataframe.utils import assert_eq
@pytest.mark.parametrize('use_dask_input', [True, False])
@pytest.mark.parametrize('npartitions', [1, 4])
@pytest.mark.parametrize('enforce_metadata', [True, False])
@pytest.mark.parametrize('transform_divisions', [True, False])
@pytest.mark.parametrize('align_dataframes', [True, False])
@pytest.mark.parametrize('overlap_setup', [(df, 0, 3), (df, 3, 0), (df, 3, 3), (df, 0, 0), (ts_constant_freq, datetime.timedelta(seconds=3), datetime.timedelta(seconds=3)), (ts_constant_freq, datetime.timedelta(seconds=3), 0)])
def test_map_overlap_multiple_dataframes(use_dask_input, npartitions, enforce_metadata, transform_divisions, align_dataframes, overlap_setup):
    dataframe, before, after = overlap_setup
    ddf = dataframe
    ddf2 = dataframe * 2
    if use_dask_input:
        ddf = dd.from_pandas(ddf, npartitions)
        ddf2 = dd.from_pandas(ddf2, 2 if align_dataframes else npartitions)

    def get_shifted_sum_arg(overlap):
        return overlap.seconds - 1 if isinstance(overlap, datetime.timedelta) else overlap
    before_shifted_sum, after_shifted_sum = (get_shifted_sum_arg(before), get_shifted_sum_arg(after))
    res = dd.map_overlap(shifted_sum, ddf, before, after, before_shifted_sum, after_shifted_sum, ddf2, align_dataframes=align_dataframes, transform_divisions=transform_divisions, enforce_metadata=enforce_metadata)
    sol = shifted_sum(dataframe, before_shifted_sum, after_shifted_sum, dataframe * 2)
    assert_eq(res, sol)
    res = dd.map_overlap(shifted_sum, ddf.b, before, after, before_shifted_sum, after_shifted_sum, ddf2.b, align_dataframes=align_dataframes, transform_divisions=transform_divisions, enforce_metadata=enforce_metadata)
    sol = shifted_sum(dataframe.b, before_shifted_sum, after_shifted_sum, dataframe.b * 2)
    assert_eq(res, sol)
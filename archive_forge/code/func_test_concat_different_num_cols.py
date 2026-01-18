import numpy as np
import pandas
import pytest
import modin.pandas as pd
from modin.config import NPartitions, StorageFormat
from modin.pandas.utils import from_pandas
from modin.utils import get_current_execution
from .utils import (
@pytest.mark.skipif(StorageFormat.get() not in ('Hdk', 'Base'), reason='https://github.com/modin-project/modin/issues/5696')
@pytest.mark.parametrize('col_type', [None, 'str'])
@pytest.mark.parametrize('df1_cols', [0, 90, 100])
@pytest.mark.parametrize('df2_cols', [0, 90, 100])
@pytest.mark.parametrize('df1_rows', [0, 100])
@pytest.mark.parametrize('df2_rows', [0, 100])
@pytest.mark.parametrize('idx_type', [None, 'str'])
@pytest.mark.parametrize('ignore_index', [True, False])
@pytest.mark.parametrize('sort', [True, False])
@pytest.mark.parametrize('join', ['inner', 'outer'])
def test_concat_different_num_cols(col_type, df1_cols, df2_cols, df1_rows, df2_rows, idx_type, ignore_index, sort, join):

    def create_frame(frame_type, ncols, nrows):

        def to_str(val):
            return f'str_{val}'
        off = 0
        data = {}
        for n in range(1, ncols + 1):
            row = range(off + 1, off + nrows + 1)
            if col_type == 'str':
                row = map(to_str, row)
            data[f'Col_{n}'] = list(row)
            off += nrows
        idx = None
        if idx_type == 'str':
            idx = pandas.Index(map(to_str, range(1, nrows + 1)), name=f'Index_{nrows}')
        df = frame_type(data=data, index=idx)
        return df

    def concat(frame_type, lib):
        df1 = create_frame(frame_type, df1_cols, df1_rows)
        df2 = create_frame(frame_type, df2_cols, df2_rows)
        return lib.concat([df1, df2], ignore_index=ignore_index, sort=sort, join=join)
    mdf = concat(pd.DataFrame, pd)
    pdf = concat(pandas.DataFrame, pandas)
    df_equals(pdf, mdf, check_dtypes=not (get_current_execution() == 'BaseOnPython' and any((o == 0 for o in (df1_cols, df2_cols, df1_rows, df2_rows)))))
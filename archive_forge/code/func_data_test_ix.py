import contextlib
from datetime import datetime
import io
import os
from pathlib import Path
import numpy as np
import pytest
from pandas.compat import IS64
from pandas.errors import EmptyDataError
import pandas.util._test_decorators as td
import pandas as pd
import pandas._testing as tm
from pandas.io.sas.sas7bdat import SAS7BDATReader
@pytest.fixture(params=[(1, range(1, 16)), (2, [16])])
def data_test_ix(request, dirpath):
    i, test_ix = request.param
    fname = os.path.join(dirpath, f'test_sas7bdat_{i}.csv')
    df = pd.read_csv(fname)
    epoch = datetime(1960, 1, 1)
    t1 = pd.to_timedelta(df['Column4'], unit='d')
    df['Column4'] = (epoch + t1).astype('M8[s]')
    t2 = pd.to_timedelta(df['Column12'], unit='d')
    df['Column12'] = (epoch + t2).astype('M8[s]')
    for k in range(df.shape[1]):
        col = df.iloc[:, k]
        if col.dtype == np.int64:
            df.isetitem(k, df.iloc[:, k].astype(np.float64))
    return (df, test_ix)
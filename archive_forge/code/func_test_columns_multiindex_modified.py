import contextlib
import datetime as dt
import hashlib
import tempfile
import time
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.io.pytables.common import (
from pandas.io.pytables import (
def test_columns_multiindex_modified(tmp_path, setup_path):
    df = DataFrame(np.random.default_rng(2).random((4, 5)), index=list('abcd'), columns=list('ABCDE'))
    df.index.name = 'letters'
    df = df.set_index(keys='E', append=True)
    data_columns = df.index.names + df.columns.tolist()
    path = tmp_path / setup_path
    df.to_hdf(path, key='df', mode='a', append=True, data_columns=data_columns, index=False)
    cols2load = list('BCD')
    cols2load_original = list(cols2load)
    read_hdf(path, 'df', columns=cols2load)
    assert cols2load_original == cols2load
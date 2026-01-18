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
def test_encoding_options(datapath):
    fname = datapath('io', 'sas', 'data', 'test1.sas7bdat')
    df1 = pd.read_sas(fname)
    df2 = pd.read_sas(fname, encoding='utf-8')
    for col in df1.columns:
        try:
            df1[col] = df1[col].str.decode('utf-8')
        except AttributeError:
            pass
    tm.assert_frame_equal(df1, df2)
    with contextlib.closing(SAS7BDATReader(fname, convert_header_text=False)) as rdr:
        df3 = rdr.read()
    for x, y in zip(df1.columns, df3.columns):
        assert x == y.decode()
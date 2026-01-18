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
def test_encoding_infer(datapath):
    fname = datapath('io', 'sas', 'data', 'test1.sas7bdat')
    with pd.read_sas(fname, encoding='infer', iterator=True) as df1_reader:
        assert df1_reader.inferred_encoding == 'cp1252'
        df1 = df1_reader.read()
    with pd.read_sas(fname, encoding='cp1252', iterator=True) as df2_reader:
        df2 = df2_reader.read()
    tm.assert_frame_equal(df1, df2)
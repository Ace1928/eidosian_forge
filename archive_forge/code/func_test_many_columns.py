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
def test_many_columns(datapath):
    fname = datapath('io', 'sas', 'data', 'many_columns.sas7bdat')
    df = pd.read_sas(fname, encoding='latin-1')
    fname = datapath('io', 'sas', 'data', 'many_columns.csv')
    df0 = pd.read_csv(fname, encoding='latin-1')
    tm.assert_frame_equal(df, df0)
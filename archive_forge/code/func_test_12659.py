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
def test_12659(datapath):
    fname = datapath('io', 'sas', 'data', 'test_12659.sas7bdat')
    df = pd.read_sas(fname)
    fname = datapath('io', 'sas', 'data', 'test_12659.csv')
    df0 = pd.read_csv(fname)
    df0 = df0.astype(np.float64)
    tm.assert_frame_equal(df, df0)
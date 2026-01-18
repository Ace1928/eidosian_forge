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
def test_0x40_control_byte(datapath):
    fname = datapath('io', 'sas', 'data', '0x40controlbyte.sas7bdat')
    df = pd.read_sas(fname, encoding='ascii')
    fname = datapath('io', 'sas', 'data', '0x40controlbyte.csv')
    df0 = pd.read_csv(fname, dtype='object')
    tm.assert_frame_equal(df, df0)
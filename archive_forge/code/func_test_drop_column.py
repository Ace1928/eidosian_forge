import bz2
import datetime as dt
from datetime import datetime
import gzip
import io
import os
import struct
import tarfile
import zipfile
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import CategoricalDtype
import pandas._testing as tm
from pandas.core.frame import (
from pandas.io.parsers import read_csv
from pandas.io.stata import (
def test_drop_column(self, datapath):
    expected = self.read_csv(datapath('io', 'data', 'stata', 'stata6.csv'))
    expected['byte_'] = expected['byte_'].astype(np.int8)
    expected['int_'] = expected['int_'].astype(np.int16)
    expected['long_'] = expected['long_'].astype(np.int32)
    expected['float_'] = expected['float_'].astype(np.float32)
    expected['double_'] = expected['double_'].astype(np.float64)
    expected['date_td'] = expected['date_td'].apply(datetime.strptime, args=('%Y-%m-%d',))
    columns = ['byte_', 'int_', 'long_']
    expected = expected[columns]
    dropped = read_stata(datapath('io', 'data', 'stata', 'stata6_117.dta'), convert_dates=True, columns=columns)
    tm.assert_frame_equal(expected, dropped)
    columns = ['int_', 'long_', 'byte_']
    expected = expected[columns]
    reordered = read_stata(datapath('io', 'data', 'stata', 'stata6_117.dta'), convert_dates=True, columns=columns)
    tm.assert_frame_equal(expected, reordered)
    msg = 'columns contains duplicate entries'
    with pytest.raises(ValueError, match=msg):
        columns = ['byte_', 'byte_']
        read_stata(datapath('io', 'data', 'stata', 'stata6_117.dta'), convert_dates=True, columns=columns)
    msg = 'The following columns were not found in the Stata data set: not_found'
    with pytest.raises(ValueError, match=msg):
        columns = ['byte_', 'int_', 'long_', 'not_found']
        read_stata(datapath('io', 'data', 'stata', 'stata6_117.dta'), convert_dates=True, columns=columns)
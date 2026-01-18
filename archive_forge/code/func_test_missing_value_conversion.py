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
@pytest.mark.parametrize('file', ['stata8_113', 'stata8_115', 'stata8_117'])
def test_missing_value_conversion(self, file, datapath):
    columns = ['int8_', 'int16_', 'int32_', 'float32_', 'float64_']
    smv = StataMissingValue(101)
    keys = sorted(smv.MISSING_VALUES.keys())
    data = []
    for i in range(27):
        row = [StataMissingValue(keys[i + j * 27]) for j in range(5)]
        data.append(row)
    expected = DataFrame(data, columns=columns)
    parsed = read_stata(datapath('io', 'data', 'stata', f'{file}.dta'), convert_missing=True)
    tm.assert_frame_equal(parsed, expected)
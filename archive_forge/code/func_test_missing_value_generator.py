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
def test_missing_value_generator(self):
    types = ('b', 'h', 'l')
    df = DataFrame([[0.0]], columns=['float_'])
    with tm.ensure_clean() as path:
        df.to_stata(path)
        with StataReader(path) as rdr:
            valid_range = rdr.VALID_RANGE
    expected_values = ['.' + chr(97 + i) for i in range(26)]
    expected_values.insert(0, '.')
    for t in types:
        offset = valid_range[t][1]
        for i in range(27):
            val = StataMissingValue(offset + 1 + i)
            assert val.string == expected_values[i]
    val = StataMissingValue(struct.unpack('<f', b'\x00\x00\x00\x7f')[0])
    assert val.string == '.'
    val = StataMissingValue(struct.unpack('<f', b'\x00\xd0\x00\x7f')[0])
    assert val.string == '.z'
    val = StataMissingValue(struct.unpack('<d', b'\x00\x00\x00\x00\x00\x00\xe0\x7f')[0])
    assert val.string == '.'
    val = StataMissingValue(struct.unpack('<d', b'\x00\x00\x00\x00\x00\x1a\xe0\x7f')[0])
    assert val.string == '.z'
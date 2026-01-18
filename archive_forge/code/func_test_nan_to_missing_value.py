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
@pytest.mark.parametrize('version', [114, 117, 118, 119, None])
def test_nan_to_missing_value(self, version):
    s1 = Series(np.arange(4.0), dtype=np.float32)
    s2 = Series(np.arange(4.0), dtype=np.float64)
    s1[::2] = np.nan
    s2[1::2] = np.nan
    original = DataFrame({'s1': s1, 's2': s2})
    original.index.name = 'index'
    with tm.ensure_clean() as path:
        original.to_stata(path, version=version)
        written_and_read_again = self.read_dta(path)
    written_and_read_again = written_and_read_again.set_index('index')
    expected = original.copy()
    expected.index = expected.index.astype(np.int32)
    tm.assert_frame_equal(written_and_read_again, expected)
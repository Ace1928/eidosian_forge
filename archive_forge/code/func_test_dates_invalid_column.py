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
def test_dates_invalid_column(self):
    original = DataFrame([datetime(2006, 11, 19, 23, 13, 20)])
    original.index.name = 'index'
    with tm.ensure_clean() as path:
        with tm.assert_produces_warning(InvalidColumnName):
            original.to_stata(path, convert_dates={0: 'tc'})
        written_and_read_again = self.read_dta(path)
    modified = original.copy()
    modified.columns = ['_0']
    modified.index = original.index.astype(np.int32)
    tm.assert_frame_equal(written_and_read_again.set_index('index'), modified)
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
def test_gzip_writing(self):
    df = DataFrame(1.1 * np.arange(120).reshape((30, 4)), columns=pd.Index(list('ABCD'), dtype=object), index=pd.Index([f'i-{i}' for i in range(30)], dtype=object))
    df.index.name = 'index'
    with tm.ensure_clean() as path:
        with gzip.GzipFile(path, 'wb') as gz:
            df.to_stata(gz, version=114)
        with gzip.GzipFile(path, 'rb') as gz:
            reread = read_stata(gz, index_col='index')
    tm.assert_frame_equal(df, reread)
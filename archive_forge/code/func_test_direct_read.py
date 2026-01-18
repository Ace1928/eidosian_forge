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
def test_direct_read(datapath, monkeypatch):
    file_path = datapath('io', 'data', 'stata', 'stata-compat-118.dta')
    with StataReader(file_path) as reader:
        assert not reader.read().empty
        assert not isinstance(reader._path_or_buf, io.BytesIO)
    with open(file_path, 'rb') as fp:
        with StataReader(fp) as reader:
            assert not reader.read().empty
            assert reader._path_or_buf is fp
    with open(file_path, 'rb') as fp:
        with io.BytesIO(fp.read()) as bio:
            with StataReader(bio) as reader:
                assert not reader.read().empty
                assert reader._path_or_buf is bio
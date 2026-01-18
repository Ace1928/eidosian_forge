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
@pytest.mark.parametrize('file', ['stata10_115', 'stata10_117'])
def test_categorical_ordering(self, file, datapath):
    file = datapath('io', 'data', 'stata', f'{file}.dta')
    parsed = read_stata(file)
    parsed_unordered = read_stata(file, order_categoricals=False)
    for col in parsed:
        if not isinstance(parsed[col].dtype, CategoricalDtype):
            continue
        assert parsed[col].cat.ordered
        assert not parsed_unordered[col].cat.ordered
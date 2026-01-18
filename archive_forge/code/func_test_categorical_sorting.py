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
@pytest.mark.parametrize('file', ['stata11_115', 'stata11_117'])
def test_categorical_sorting(self, file, datapath):
    parsed = read_stata(datapath('io', 'data', 'stata', f'{file}.dta'))
    parsed = parsed.sort_values('srh', na_position='first')
    parsed.index = pd.RangeIndex(len(parsed))
    codes = [-1, -1, 0, 1, 1, 1, 2, 2, 3, 4]
    categories = ['Poor', 'Fair', 'Good', 'Very good', 'Excellent']
    cat = pd.Categorical.from_codes(codes=codes, categories=categories, ordered=True)
    expected = Series(cat, name='srh')
    tm.assert_series_equal(expected, parsed['srh'])
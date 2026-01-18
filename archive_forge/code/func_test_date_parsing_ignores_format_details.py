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
@pytest.mark.parametrize('column', ['ms', 'day', 'week', 'month', 'qtr', 'half', 'yr'])
def test_date_parsing_ignores_format_details(self, column, datapath):
    df = read_stata(datapath('io', 'data', 'stata', 'stata13_dates.dta'))
    unformatted = df.loc[0, column]
    formatted = df.loc[0, column + '_fmt']
    assert unformatted == formatted
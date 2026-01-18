import contextlib
import csv
import inspect
import os
import sys
import unittest.mock as mock
from collections import defaultdict
from io import BytesIO, StringIO
from pathlib import Path
from typing import Dict
import fastparquet
import numpy as np
import pandas
import pandas._libs.lib as lib
import pyarrow as pa
import pyarrow.dataset
import pytest
import sqlalchemy as sa
from packaging import version
from pandas._testing import ensure_clean
from pandas.errors import ParserWarning
from scipy import sparse
from modin.config import (
from modin.db_conn import ModinDatabaseConnection, UnsupportedDatabaseException
from modin.pandas.io import from_arrow, from_dask, from_ray, to_pandas
from modin.tests.test_utils import warns_that_defaulting_to_pandas
from .utils import (
from .utils import test_data as utils_test_data
from .utils import time_parsing_csv_path
from modin.config import NPartitions
@pytest.mark.parametrize('kwargs', [{'colspecs': [(0, 11), (11, 15), (19, 24), (27, 32), (35, 40), (43, 48), (51, 56), (59, 64), (67, 72), (75, 80), (83, 88), (91, 96), (99, 104), (107, 112)], 'names': ['stationID', 'year', 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 'na_values': ['-9999'], 'index_col': ['stationID', 'year']}, {'widths': [20, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], 'names': ['id', 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 'index_col': [0]}])
def test_fwf_file_colspecs_widths(self, make_fwf_file, kwargs):
    unique_filename = make_fwf_file()
    modin_df = pd.read_fwf(unique_filename, **kwargs)
    pandas_df = pd.read_fwf(unique_filename, **kwargs)
    df_equals(modin_df, pandas_df)
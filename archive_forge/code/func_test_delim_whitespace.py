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
@pytest.mark.parametrize('delim_whitespace', [True, False])
def test_delim_whitespace(self, delim_whitespace, tmp_path):
    if StorageFormat.get() == 'Hdk' and delim_whitespace:
        pytest.xfail(reason='https://github.com/modin-project/modin/issues/6999')
    str_delim_whitespaces = 'col1 col2  col3   col4\n5 6   7  8\n9  10    11 12\n'
    unique_filename = get_unique_filename(data_dir=tmp_path)
    eval_io_from_str(str_delim_whitespaces, unique_filename, delim_whitespace=delim_whitespace)
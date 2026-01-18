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
def test_fwf_file_parse_dates(self, make_fwf_file):
    dates = pandas.date_range('2000', freq='h', periods=10)
    fwf_data = 'col1 col2        col3 col4'
    for i in range(10, 20):
        fwf_data = fwf_data + '\n{col1}   {col2}  {col3}   {col4}'.format(col1=str(i), col2=str(dates[i - 10].date()), col3=str(i), col4=str(dates[i - 10].time()))
    unique_filename = make_fwf_file(fwf_data=fwf_data)
    eval_io(fn_name='read_fwf', filepath_or_buffer=unique_filename, parse_dates=[['col2', 'col4']])
    eval_io(fn_name='read_fwf', filepath_or_buffer=unique_filename, parse_dates={'time': ['col2', 'col4']})
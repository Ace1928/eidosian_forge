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
def test_HDFStore(self, tmp_path):
    unique_filename_modin = get_unique_filename(extension='hdf', data_dir=tmp_path)
    unique_filename_pandas = get_unique_filename(extension='hdf', data_dir=tmp_path)
    modin_store = pd.HDFStore(unique_filename_modin)
    pandas_store = pandas.HDFStore(unique_filename_pandas)
    modin_df, pandas_df = create_test_dfs(TEST_DATA)
    modin_store['foo'] = modin_df
    pandas_store['foo'] = pandas_df
    modin_df = modin_store.get('foo')
    pandas_df = pandas_store.get('foo')
    df_equals(modin_df, pandas_df)
    modin_store.close()
    pandas_store.close()
    modin_df = pandas.read_hdf(unique_filename_modin, key='foo', mode='r')
    pandas_df = pandas.read_hdf(unique_filename_pandas, key='foo', mode='r')
    df_equals(modin_df, pandas_df)
    assert isinstance(modin_store, pd.HDFStore)
    with ensure_clean('.hdf5') as hdf_file:
        with pd.HDFStore(hdf_file, mode='w') as store:
            store.append('data/df1', pd.DataFrame(np.random.randn(5, 5)))
            store.append('data/df2', pd.DataFrame(np.random.randn(4, 4)))
        modin_df = pd.read_hdf(hdf_file, key='data/df1', mode='r')
        pandas_df = pandas.read_hdf(hdf_file, key='data/df1', mode='r')
    df_equals(modin_df, pandas_df)
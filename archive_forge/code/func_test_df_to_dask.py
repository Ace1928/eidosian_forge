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
@pytest.mark.skipif(condition=Engine.get() != 'Dask', reason='Modin DataFrame can only be converted to a Dask DataFrame if Modin uses a Dask engine.')
@pytest.mark.filterwarnings(default_to_pandas_ignore_string)
def test_df_to_dask():
    index = pandas.DatetimeIndex(pandas.date_range('2000', freq='h', periods=len(TEST_DATA['col1'])))
    modin_df, pandas_df = create_test_dfs(TEST_DATA, index=index)
    dask_df = modin_df.modin.to_dask()
    df_equals(dask_df.compute(), pandas_df)
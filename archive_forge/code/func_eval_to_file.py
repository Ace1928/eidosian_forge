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
def eval_to_file(tmp_dir, modin_obj, pandas_obj, fn, extension, **fn_kwargs):
    """
    Test `fn` method of `modin_obj` and `pandas_obj`.

    Parameters
    ----------
    tmp_dir : Union[str, Path]
        Temporary directory.
    modin_obj: Modin DataFrame or Series
        Object to test.
    pandas_obj: Pandas DataFrame or Series
        Object to test.
    fn: str
        Name of the method, that should be tested.
    extension: str
        Extension of the test file.
    """
    unique_filename_modin = get_unique_filename(extension=extension, data_dir=tmp_dir)
    unique_filename_pandas = get_unique_filename(extension=extension, data_dir=tmp_dir)
    last_exception = None
    for _ in range(3):
        try:
            getattr(modin_obj, fn)(unique_filename_modin, **fn_kwargs)
        except EXCEPTIONS as err:
            last_exception = err
            continue
        break
    if last_exception:
        raise last_exception
    getattr(pandas_obj, fn)(unique_filename_pandas, **fn_kwargs)
    assert assert_files_eq(unique_filename_modin, unique_filename_pandas)
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
def eval_to_csv_file(tmp_dir, modin_obj, pandas_obj, extension, **kwargs):
    if extension is None:
        kwargs['mode'] = 't'
        kwargs['compression'] = 'infer'
        modin_csv = modin_obj.to_csv(**kwargs)
        pandas_csv = pandas_obj.to_csv(**kwargs)
        if modin_csv == pandas_csv:
            return
        force_read = True
        modin_file = get_unique_filename(extension='csv', data_dir=tmp_dir)
        pandas_file = get_unique_filename(extension='csv', data_dir=tmp_dir)
        with open(modin_file, 'w') as file:
            file.write(modin_csv)
        with open(pandas_file, 'w') as file:
            file.write(pandas_csv)
    else:
        force_read = extension != 'csv' or kwargs.get('compression', None)
        modin_file = get_unique_filename(extension=extension, data_dir=tmp_dir)
        pandas_file = get_unique_filename(extension=extension, data_dir=tmp_dir)
        modin_obj.to_csv(modin_file, **kwargs)
        pandas_obj.to_csv(pandas_file, **kwargs)
    if force_read or not assert_files_eq(modin_file, pandas_file):
        read_kwargs = {}
        if kwargs.get('index', None) is not False:
            read_kwargs['index_col'] = 0
        if (value := kwargs.get('sep', None)) is not None:
            read_kwargs['sep'] = value
        if (value := kwargs.get('compression', None)) is not None:
            read_kwargs['compression'] = value
        modin_obj = pandas.read_csv(modin_file, **read_kwargs)
        pandas_obj = pandas.read_csv(pandas_file, **read_kwargs)
        df_equals(pandas_obj, modin_obj)
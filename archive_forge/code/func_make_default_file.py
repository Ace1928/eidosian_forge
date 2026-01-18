import csv
import functools
import itertools
import math
import os
import re
from io import BytesIO
from pathlib import Path
from string import ascii_letters
from typing import Union
import numpy as np
import pandas
import psutil
import pytest
from pandas.core.dtypes.common import (
import modin.pandas as pd
from modin.config import (
from modin.pandas.io import to_pandas
from modin.pandas.testing import (
from modin.utils import try_cast_to_pandas
def make_default_file(file_type: str, data_dir: str):
    """Helper function for pytest fixtures."""

    def _create_file(filename, force, nrows, ncols, func: str, func_kw=None):
        """
        Helper function that creates a dataframe before writing it to a file.

        Eliminates the duplicate code that is needed before of output functions calls.

        Notes
        -----
        Importantly, names of created files are added to `filenames` variable for
        their further automatic deletion. Without this step, files created by
        `pytest` fixtures will not be deleted.
        """
        if force or not os.path.exists(filename):
            df = pandas.DataFrame({f'col{x + 1}': np.arange(nrows) for x in range(ncols)})
            getattr(df, func)(filename, **func_kw if func_kw else {})
    file_type_to_extension = {'excel': 'xlsx', 'fwf': 'txt', 'pickle': 'pkl'}
    extension = file_type_to_extension.get(file_type, file_type)

    def _make_default_file(nrows=NROWS, ncols=2, force=True, **kwargs):
        filename = get_unique_filename(extension=extension, data_dir=data_dir)
        if file_type == 'json':
            lines = kwargs.get('lines')
            func_kw = {'lines': lines, 'orient': 'records'} if lines else {}
            _create_file(filename, force, nrows, ncols, 'to_json', func_kw)
        elif file_type in ('html', 'excel', 'feather', 'stata', 'pickle'):
            _create_file(filename, force, nrows, ncols, f'to_{file_type}')
        elif file_type == 'hdf':
            func_kw = {'key': 'df', 'format': kwargs.get('format')}
            _create_file(filename, force, nrows, ncols, 'to_hdf', func_kw)
        elif file_type == 'fwf':
            if force or not os.path.exists(filename):
                fwf_data = kwargs.get('fwf_data')
                if fwf_data is None:
                    with open('modin/tests/pandas/data/test_data.fwf', 'r') as fwf_file:
                        fwf_data = fwf_file.read()
                with open(filename, 'w') as f:
                    f.write(fwf_data)
        else:
            raise ValueError(f'Unsupported file type: {file_type}')
        return filename
    return _make_default_file
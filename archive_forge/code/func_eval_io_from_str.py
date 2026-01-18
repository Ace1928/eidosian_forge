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
def eval_io_from_str(csv_str: str, unique_filename: str, **kwargs):
    """Evaluate I/O operation outputs equality check by using `csv_str`
    data passed as python str (csv test file will be created from `csv_str`).

    Parameters
    ----------
    csv_str: str
        Test data for storing to csv file.
    unique_filename: str
        csv file name.
    """
    with open(unique_filename, 'w') as f:
        f.write(csv_str)
    eval_io(filepath_or_buffer=unique_filename, fn_name='read_csv', **kwargs)
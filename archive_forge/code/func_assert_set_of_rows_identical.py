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
def assert_set_of_rows_identical(df1, df2):
    """
    Assert that the set of rows for the passed dataframes is identical.

    Works much slower than ``df1.equals(df2)``, so it's recommended to use this
    function only in exceptional cases.
    """
    df1, df2 = map(lambda df: (df.to_frame() if df.ndim == 1 else df).replace({np.nan: None}), (df1, df2))
    rows1 = set(((idx, *row.tolist()) for idx, row in df1.iterrows()))
    rows2 = set(((idx, *row.tolist()) for idx, row in df2.iterrows()))
    assert rows1 == rows2
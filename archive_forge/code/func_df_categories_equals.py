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
def df_categories_equals(df1, df2):
    if not hasattr(df1, 'select_dtypes'):
        if isinstance(df1, pandas.CategoricalDtype):
            categories_equals(df1, df2)
        elif isinstance(getattr(df1, 'dtype'), pandas.CategoricalDtype) and isinstance(getattr(df2, 'dtype'), pandas.CategoricalDtype):
            categories_equals(df1.dtype, df2.dtype)
        return True
    df1_categorical = df1.select_dtypes(include='category')
    df2_categorical = df2.select_dtypes(include='category')
    assert df1_categorical.columns.equals(df2_categorical.columns)
    for i in range(len(df1_categorical.columns)):
        assert_extension_array_equal(df1_categorical.iloc[:, i].values, df2_categorical.iloc[:, i].values, check_dtype=False)
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
def assert_dtypes_equal(df1, df2):
    """
    Assert that the two passed DataFrame/Series objects have equal dtypes.

    The function doesn't require that the dtypes are identical, it has the following reliefs:
        1. The dtypes are not required to be in the same order
           (e.g. {"col1": int, "col2": float} == {"col2": float, "col1": int})
        2. The dtypes are only required to be in the same class
           (e.g. both numerical, both categorical, etc...)

    Parameters
    ----------
    df1 : DataFrame or Series
    df2 : DataFrame or Series
    """
    if not isinstance(df1, (pandas.Series, pd.Series, pandas.DataFrame, pd.DataFrame)) or not isinstance(df2, (pandas.Series, pd.Series, pandas.DataFrame, pd.DataFrame)):
        return
    if isinstance(df1.dtypes, (pandas.Series, pd.Series)):
        dtypes1 = df1.dtypes
        dtypes2 = df2.dtypes
    else:
        dtypes1 = pandas.Series({'col': df1.dtypes})
        dtypes2 = pandas.Series({'col': df2.dtypes})
    assert len(dtypes1.index.difference(dtypes2.index)) == 0
    assert len(dtypes1) == len(dtypes2)
    dtype_comparators = (is_numeric_dtype, lambda obj: is_object_dtype(obj) or is_string_dtype(obj), is_bool_dtype, lambda obj: isinstance(obj, pandas.CategoricalDtype), is_datetime64_any_dtype, is_timedelta64_dtype, lambda obj: isinstance(obj, pandas.PeriodDtype))
    for col in dtypes1.keys():
        for comparator in dtype_comparators:
            if assert_all_act_same(comparator, dtypes1[col], dtypes2[col]):
                break
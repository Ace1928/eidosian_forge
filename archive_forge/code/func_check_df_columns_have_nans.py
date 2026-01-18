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
def check_df_columns_have_nans(df, cols):
    """Checks if there are NaN values in specified columns of a dataframe.

    :param df: Dataframe to check.
    :param cols: One column name or list of column names.
    :return:
        True if specified columns of dataframe contains NaNs.
    """
    return pandas.api.types.is_list_like(cols) and (any((isinstance(x, str) and x in df.columns and df[x].hasnans for x in cols)) or any((isinstance(x, pd.Series) and x._parent is df and x.hasnans for x in cols))) or (not pandas.api.types.is_list_like(cols) and cols in df.columns and df[cols].hasnans)
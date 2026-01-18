import warnings
import matplotlib
import numpy as np
import pandas
import pytest
import modin.pandas as pd
from modin.config import Engine, NPartitions, RangePartitioning, StorageFormat
from modin.pandas.io import to_pandas
from modin.tests.pandas.utils import (
from modin.tests.test_utils import warns_that_defaulting_to_pandas
def df_equals_and_sort(df1, df2):
    """Sort dataframe's rows and run ``df_equals()`` for them."""
    df1 = df1.sort_values(by=df1.columns.tolist(), ignore_index=True)
    df2 = df2.sort_values(by=df2.columns.tolist(), ignore_index=True)
    df_equals(df1, df2)
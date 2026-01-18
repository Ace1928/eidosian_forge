import os
import sys
import matplotlib
import numpy as np
import pandas
import pytest
from pandas._testing import ensure_clean
import modin.pandas as pd
from modin.config import MinPartitionSize, NPartitions, StorageFormat
from modin.pandas.indexing import is_range_like
from modin.pandas.testing import assert_index_equal
from modin.tests.pandas.utils import (
from modin.utils import get_current_execution
def test___setitem__assigning_single_categorical_sets_correct_dtypes():
    modin_df = pd.DataFrame({'categories': ['A']})
    modin_df['categories'] = pd.Categorical(['A'])
    pandas_df = pandas.DataFrame({'categories': ['A']})
    pandas_df['categories'] = pandas.Categorical(['A'])
    df_equals(modin_df, pandas_df)
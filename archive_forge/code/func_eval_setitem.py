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
def eval_setitem(md_df, pd_df, value, col=None, loc=None, expected_exception=None):
    if loc is not None:
        col = pd_df.columns[loc]
    value_getter = value if callable(value) else lambda *args, **kwargs: value
    eval_general(md_df, pd_df, lambda df: df.__setitem__(col, value_getter(df)), __inplace__=True, expected_exception=expected_exception)
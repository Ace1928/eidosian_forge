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
@pytest.mark.parametrize('lookup', [[60, 70, 90], [60.5, 70.5, 100]])
@pytest.mark.parametrize('subset', ['col2', 'col1', ['col1', 'col2'], None])
def test_asof_large(lookup, subset):
    data = test_data['float_nan_data']
    index = list(range(NROWS))
    modin_where = pd.Index(lookup)
    pandas_where = pandas.Index(lookup)
    compare_asof(data, index, modin_where, pandas_where, subset)
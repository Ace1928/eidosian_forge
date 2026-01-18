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
@pytest.fixture
def loc_iter_dfs():
    columns = ['col1', 'col2', 'col3']
    index = ['row1', 'row2', 'row3']
    return create_test_dfs({col: [idx] * len(index) for idx, col in enumerate(columns)}, columns=columns, index=index)
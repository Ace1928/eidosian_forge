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
def test_loc_multi_index_level_two_has_same_name_as_column():
    eval_general(*create_test_dfs(pandas.DataFrame([[0]], index=[pd.Index(['foo']), pd.Index(['bar'])], columns=['bar'])), lambda df: df.loc['foo', 'bar'])
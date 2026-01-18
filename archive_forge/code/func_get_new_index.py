import io
import warnings
import matplotlib
import numpy as np
import pandas
import pandas._libs.lib as lib
import pytest
from numpy.testing import assert_array_equal
import modin.pandas as pd
from modin.config import Engine, NPartitions, StorageFormat
from modin.pandas.io import to_pandas
from modin.tests.pandas.utils import (
from modin.tests.test_utils import warns_that_defaulting_to_pandas
from modin.utils import get_current_execution
def get_new_index(index, cond):
    if cond == 'col_multi_tree' or cond == 'idx_multi_tree':
        return generate_multiindex(len(index), nlevels=3, is_tree_like=True)
    elif cond == 'col_multi_not_tree' or cond == 'idx_multi_not_tree':
        return generate_multiindex(len(index), nlevels=3)
    else:
        return index
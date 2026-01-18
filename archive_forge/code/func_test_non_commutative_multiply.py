import matplotlib
import numpy as np
import pandas
import pytest
import modin.pandas as pd
from modin.config import Engine, NPartitions, StorageFormat
from modin.core.dataframe.pandas.partitioning.axis_partition import (
from modin.tests.pandas.utils import (
from modin.tests.test_utils import warns_that_defaulting_to_pandas
from modin.utils import get_current_execution
def test_non_commutative_multiply():
    modin_df, pandas_df = create_test_dfs([1], dtype=int)
    integer = NonCommutativeMultiplyInteger(2)
    eval_general(modin_df, pandas_df, lambda s: integer * s)
    eval_general(modin_df, pandas_df, lambda s: s * integer)
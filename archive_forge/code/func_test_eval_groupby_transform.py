import matplotlib
import numpy as np
import pandas
import pytest
from pandas.core.dtypes.common import is_list_like
import modin.pandas as pd
from modin.config import MinPartitionSize, NPartitions
from modin.tests.pandas.utils import (
from modin.tests.test_utils import warns_that_defaulting_to_pandas
from modin.utils import get_current_execution
def test_eval_groupby_transform():
    df = pd.DataFrame({'num': range(1, 1001), 'group': ['A'] * 500 + ['B'] * 500})
    assert df.eval("num.groupby(group).transform('min')").unique().tolist() == [1, 501]
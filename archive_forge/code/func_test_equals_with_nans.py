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
def test_equals_with_nans():
    df1 = pd.DataFrame([0, 1, None], dtype='uint8[pyarrow]')
    df2 = pd.DataFrame([None, None, None], dtype='uint8[pyarrow]')
    assert not df1.equals(df2)
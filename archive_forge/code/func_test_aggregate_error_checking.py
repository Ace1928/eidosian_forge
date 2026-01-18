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
def test_aggregate_error_checking():
    modin_df = pd.DataFrame(test_data['float_nan_data'])
    with warns_that_defaulting_to_pandas():
        modin_df.aggregate({modin_df.columns[0]: 'sum', modin_df.columns[1]: 'mean'})
    with warns_that_defaulting_to_pandas():
        modin_df.aggregate('cumproduct')
    with pytest.raises(ValueError):
        modin_df.aggregate('NOT_EXISTS')
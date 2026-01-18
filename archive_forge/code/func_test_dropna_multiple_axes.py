import matplotlib
import numpy as np
import pandas
import pytest
import modin.pandas as pd
from modin.config import MinPartitionSize, NPartitions, StorageFormat
from modin.core.dataframe.pandas.metadata import LazyProxyCategoricalDtype
from modin.core.storage_formats.pandas.utils import split_result_of_axis_func_pandas
from modin.pandas.testing import assert_index_equal, assert_series_equal
from modin.tests.pandas.utils import (
from modin.tests.test_utils import warns_that_defaulting_to_pandas
from modin.utils import get_current_execution
@pytest.mark.parametrize('data', test_data_values, ids=test_data_keys)
def test_dropna_multiple_axes(data):
    modin_df = pd.DataFrame(data)
    with pytest.raises(TypeError):
        modin_df.dropna(how='all', axis=[0, 1])
    with pytest.raises(TypeError):
        modin_df.dropna(how='all', axis=(0, 1))
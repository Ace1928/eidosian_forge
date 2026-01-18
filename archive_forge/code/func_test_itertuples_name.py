import io
import warnings
import matplotlib
import numpy as np
import pandas
import pytest
import modin.pandas as pd
from modin.config import NPartitions
from modin.pandas.utils import SET_DATAFRAME_ATTRIBUTE_WARNING
from modin.tests.pandas.utils import (
from modin.tests.test_utils import warns_that_defaulting_to_pandas
@pytest.mark.parametrize('name', [None, 'NotPandas'])
def test_itertuples_name(name):
    data = test_data['float_nan_data']
    modin_df, pandas_df = (pd.DataFrame(data), pandas.DataFrame(data))
    modin_it_custom = modin_df.itertuples(name=name)
    pandas_it_custom = pandas_df.itertuples(name=name)
    for modin_row, pandas_row in zip(modin_it_custom, pandas_it_custom):
        np.testing.assert_equal(modin_row, pandas_row)
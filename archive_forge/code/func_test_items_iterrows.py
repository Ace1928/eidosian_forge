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
@pytest.mark.parametrize('method', ['items', 'iterrows'])
def test_items_iterrows(method):
    data = test_data['float_nan_data']
    modin_df, pandas_df = (pd.DataFrame(data), pandas.DataFrame(data))
    for modin_item, pandas_item in zip(getattr(modin_df, method)(), getattr(pandas_df, method)()):
        modin_index, modin_series = modin_item
        pandas_index, pandas_series = pandas_item
        df_equals(pandas_series, modin_series)
        assert pandas_index == modin_index
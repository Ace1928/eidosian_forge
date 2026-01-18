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
@pytest.mark.parametrize('data', test_data_values, ids=test_data_keys)
def test___copy__(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)
    modin_df_copy, pandas_df_copy = (modin_df.__copy__(), pandas_df.__copy__())
    df_equals(modin_df_copy, pandas_df_copy)
import numpy as np
import pandas
import pytest
import modin.pandas as pd
from modin.config import NPartitions, StorageFormat
from modin.pandas.utils import from_pandas
from modin.utils import get_current_execution
from .utils import (
def test_concat_empty_df_series():
    pdf = pandas.concat((pandas.DataFrame({'A': [1, 2, 3]}), pandas.Series()))
    mdf = pd.concat((pd.DataFrame({'A': [1, 2, 3]}), pd.Series()))
    df_equals(pdf, mdf, check_dtypes=False)
    pdf = pandas.concat((pandas.DataFrame(), pandas.Series([1, 2, 3])))
    mdf = pd.concat((pd.DataFrame(), pd.Series([1, 2, 3])))
    df_equals(pdf, mdf, check_dtypes=False)
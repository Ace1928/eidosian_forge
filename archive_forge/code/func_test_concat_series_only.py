import numpy as np
import pandas
import pytest
import modin.pandas as pd
from modin.config import NPartitions, StorageFormat
from modin.pandas.utils import from_pandas
from modin.utils import get_current_execution
from .utils import (
def test_concat_series_only():
    modin_series = pd.Series(list(range(1000)))
    pandas_series = pandas.Series(list(range(1000)))
    df_equals(pd.concat([modin_series, modin_series]), pandas.concat([pandas_series, pandas_series]))
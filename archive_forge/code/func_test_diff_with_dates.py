from __future__ import annotations
import datetime
import itertools
import json
import unittest.mock as mock
import matplotlib
import numpy as np
import pandas
import pandas._libs.lib as lib
import pytest
from numpy.testing import assert_array_equal
from pandas.core.indexing import IndexingError
from pandas.errors import SpecificationError
import modin.pandas as pd
from modin.config import Engine, NPartitions, StorageFormat
from modin.pandas.io import to_pandas
from modin.pandas.testing import assert_series_equal
from modin.tests.test_utils import warns_that_defaulting_to_pandas
from modin.utils import get_current_execution, try_cast_to_pandas
from .utils import (
def test_diff_with_dates():
    data = pandas.date_range('2018-01-01', periods=15, freq='h').values
    pandas_series = pandas.Series(data)
    modin_series = pd.Series(pandas_series)
    pandas_result = pandas_series.diff()
    modin_result = modin_series.diff()
    df_equals(modin_result, pandas_result)
    td_pandas_result = pandas_result.diff()
    td_modin_result = modin_result.diff()
    df_equals(td_modin_result, td_pandas_result)
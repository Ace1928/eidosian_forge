import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.apply.common import series_transform_kernels
def test_agg_evaluate_lambdas(string_series):
    with tm.assert_produces_warning(FutureWarning):
        result = string_series.agg(lambda x: type(x))
    assert isinstance(result, Series) and len(result) == len(string_series)
    with tm.assert_produces_warning(FutureWarning):
        result = string_series.agg(type)
    assert isinstance(result, Series) and len(result) == len(string_series)
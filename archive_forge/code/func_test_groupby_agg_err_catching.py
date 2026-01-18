import datetime as dt
from functools import partial
import numpy as np
import pytest
from pandas.errors import SpecificationError
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.io.formats.printing import pprint_thing
@pytest.mark.parametrize('err_cls', [NotImplementedError, RuntimeError, KeyError, IndexError, OSError, ValueError, ArithmeticError, AttributeError])
def test_groupby_agg_err_catching(err_cls):
    from pandas.tests.extension.decimal.array import DecimalArray, make_data, to_decimal
    data = make_data()[:5]
    df = DataFrame({'id1': [0, 0, 0, 1, 1], 'id2': [0, 1, 0, 1, 1], 'decimals': DecimalArray(data)})
    expected = Series(to_decimal([data[0], data[3]]))

    def weird_func(x):
        if len(x) == 0:
            raise err_cls
        return x.iloc[0]
    result = df['decimals'].groupby(df['id1']).agg(weird_func)
    tm.assert_series_equal(result, expected, check_names=False)
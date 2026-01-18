import datetime as dt
from functools import partial
import numpy as np
import pytest
from pandas.errors import SpecificationError
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.io.formats.printing import pprint_thing
def test_agg_partial_failure_raises():
    df = DataFrame({'data1': np.random.default_rng(2).standard_normal(5), 'data2': np.random.default_rng(2).standard_normal(5), 'key1': ['a', 'a', 'b', 'b', 'a'], 'key2': ['one', 'two', 'one', 'two', 'one']})
    grouped = df.groupby('key1')

    def peak_to_peak(arr):
        return arr.max() - arr.min()
    with pytest.raises(TypeError, match='unsupported operand type'):
        grouped.agg([peak_to_peak])
    with pytest.raises(TypeError, match='unsupported operand type'):
        grouped.agg(peak_to_peak)
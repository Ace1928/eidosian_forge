import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_crosstab_pass_values(self):
    a = np.random.default_rng(2).integers(0, 7, size=100)
    b = np.random.default_rng(2).integers(0, 3, size=100)
    c = np.random.default_rng(2).integers(0, 5, size=100)
    values = np.random.default_rng(2).standard_normal(100)
    table = crosstab([a, b], c, values, aggfunc='sum', rownames=['foo', 'bar'], colnames=['baz'])
    df = DataFrame({'foo': a, 'bar': b, 'baz': c, 'values': values})
    expected = df.pivot_table('values', index=['foo', 'bar'], columns='baz', aggfunc='sum')
    tm.assert_frame_equal(table, expected)
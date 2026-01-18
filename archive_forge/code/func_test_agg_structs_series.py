import datetime as dt
from functools import partial
import numpy as np
import pytest
from pandas.errors import SpecificationError
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.io.formats.printing import pprint_thing
@pytest.mark.parametrize('structure, expected', [(tuple, Series([(1, 1, 1), (3, 4, 4)], index=[1, 3], name='C')), (list, Series([[1, 1, 1], [3, 4, 4]], index=[1, 3], name='C')), (lambda x: tuple(x), Series([(1, 1, 1), (3, 4, 4)], index=[1, 3], name='C')), (lambda x: list(x), Series([[1, 1, 1], [3, 4, 4]], index=[1, 3], name='C'))])
def test_agg_structs_series(structure, expected):
    df = DataFrame({'A': [1, 1, 1, 3, 3, 3], 'B': [1, 1, 1, 4, 4, 4], 'C': [1, 1, 1, 3, 4, 4]})
    result = df.groupby('A')['C'].aggregate(structure)
    expected.index.name = 'A'
    tm.assert_series_equal(result, expected)
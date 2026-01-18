import datetime as dt
from functools import partial
import numpy as np
import pytest
from pandas.errors import SpecificationError
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.io.formats.printing import pprint_thing
@pytest.mark.parametrize('structure, expected', [(tuple, DataFrame({'C': {(1, 1): (1, 1, 1), (3, 4): (3, 4, 4)}})), (list, DataFrame({'C': {(1, 1): [1, 1, 1], (3, 4): [3, 4, 4]}})), (lambda x: tuple(x), DataFrame({'C': {(1, 1): (1, 1, 1), (3, 4): (3, 4, 4)}})), (lambda x: list(x), DataFrame({'C': {(1, 1): [1, 1, 1], (3, 4): [3, 4, 4]}}))])
def test_agg_structs_dataframe(structure, expected):
    df = DataFrame({'A': [1, 1, 1, 3, 3, 3], 'B': [1, 1, 1, 4, 4, 4], 'C': [1, 1, 1, 3, 4, 4]})
    result = df.groupby(['A', 'B']).aggregate(structure)
    expected.index.names = ['A', 'B']
    tm.assert_frame_equal(result, expected)
import datetime as dt
from functools import partial
import numpy as np
import pytest
from pandas.errors import SpecificationError
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.io.formats.printing import pprint_thing
def test_agg_list_like_func():
    df = DataFrame({'A': [str(x) for x in range(3)], 'B': [str(x) for x in range(3)]})
    grouped = df.groupby('A', as_index=False, sort=False)
    result = grouped.agg({'B': lambda x: list(x)})
    expected = DataFrame({'A': [str(x) for x in range(3)], 'B': [[str(x)] for x in range(3)]})
    tm.assert_frame_equal(result, expected)
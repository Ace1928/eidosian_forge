from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.groupby.grouper import Grouping
def test_gb_key_len_equal_axis_len(self):
    df = DataFrame([['foo', 'bar', 'B', 1], ['foo', 'bar', 'B', 2], ['foo', 'baz', 'C', 3]], columns=['first', 'second', 'third', 'one'])
    df = df.set_index(['first', 'second'])
    df = df.groupby(['first', 'second', 'third']).size()
    assert df.loc['foo', 'bar', 'B'] == 2
    assert df.loc['foo', 'baz', 'C'] == 1
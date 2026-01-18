from string import ascii_letters
import numpy as np
import pytest
from pandas.errors import (
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.arm_slow
def test_detect_chained_assignment_false_positives(self):
    df = DataFrame({'column1': ['a', 'a', 'a'], 'column2': [4, 8, 9]})
    str(df)
    df['column1'] = df['column1'] + 'b'
    str(df)
    df = df[df['column2'] != 8]
    str(df)
    df['column1'] = df['column1'] + 'c'
    str(df)
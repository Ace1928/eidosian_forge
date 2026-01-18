import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_format_integer_names():
    index = MultiIndex(levels=[[0, 1], [0, 1]], codes=[[0, 0, 1, 1], [0, 1, 0, 1]], names=[0, 1])
    msg = 'MultiIndex.format is deprecated'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        index.format(names=True)
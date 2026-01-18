import re
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('col_level', [0, 'CAP'])
def test_col_level(self, col_level, df1):
    res = df1.melt(col_level=col_level)
    assert res.columns.tolist() == ['CAP', 'value']
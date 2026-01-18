import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_multiindex_long_element(self):
    data = MultiIndex.from_tuples([('c' * 62,)])
    expected = "MultiIndex([('cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc',)],\n           )"
    assert str(data) == expected
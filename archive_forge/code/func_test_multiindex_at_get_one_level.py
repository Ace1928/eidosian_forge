from datetime import (
import itertools
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_multiindex_at_get_one_level(self):
    s2 = Series((0, 1), index=[[False, True]])
    result = s2.at[False]
    assert result == 0
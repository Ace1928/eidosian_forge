from datetime import timedelta
import re
import numpy as np
import pytest
from pandas._libs import index as libindex
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_get_loc_duplicates(self):
    index = Index([2, 2, 2, 2])
    result = index.get_loc(2)
    expected = slice(0, 4)
    assert result == expected
    index = Index(['c', 'a', 'a', 'b', 'b'])
    rs = index.get_loc('c')
    xp = 0
    assert rs == xp
    with pytest.raises(KeyError, match='2'):
        index.get_loc(2)
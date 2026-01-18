from itertools import product
import numpy as np
import pytest
from pandas._libs import (
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('names', [['a', 'b', 'a'], [1, 1, 2], [1, 'a', 1]])
def test_duplicate_level_names(names):
    mi = MultiIndex.from_product([[0, 1]] * 3, names=names)
    assert mi.names == names
    mi = MultiIndex.from_product([[0, 1]] * 3)
    mi = mi.rename(names)
    assert mi.names == names
    mi.rename(names[1], level=1, inplace=True)
    mi = mi.rename([names[0], names[2]], level=[0, 2])
    assert mi.names == names
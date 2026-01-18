from datetime import (
import itertools
import numpy as np
import pytest
from pandas.core.dtypes.cast import construct_1d_object_array_from_listlike
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_constructor_nonhashable_names():
    levels = [[1, 2], ['one', 'two']]
    codes = [[0, 0, 1, 1], [0, 1, 0, 1]]
    names = (['foo'], ['bar'])
    msg = 'MultiIndex\\.name must be a hashable type'
    with pytest.raises(TypeError, match=msg):
        MultiIndex(levels=levels, codes=codes, names=names)
    mi = MultiIndex(levels=[[1, 2], ['one', 'two']], codes=[[0, 0, 1, 1], [0, 1, 0, 1]], names=('foo', 'bar'))
    renamed = [['fooo'], ['barr']]
    with pytest.raises(TypeError, match=msg):
        mi.rename(names=renamed)
    with pytest.raises(TypeError, match=msg):
        mi.set_names(names=renamed)
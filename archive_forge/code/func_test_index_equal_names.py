import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('name1,name2', [(None, 'x'), ('x', 'x'), (np.nan, np.nan), (NaT, NaT), (np.nan, NaT)])
def test_index_equal_names(name1, name2):
    idx1 = Index([1, 2, 3], name=name1)
    idx2 = Index([1, 2, 3], name=name2)
    if name1 == name2 or name1 is name2:
        tm.assert_index_equal(idx1, idx2)
    else:
        name1 = "'x'" if name1 == 'x' else name1
        name2 = "'x'" if name2 == 'x' else name2
        msg = f'Index are different\n\nAttribute "names" are different\n\\[left\\]:  \\[{name1}\\]\n\\[right\\]: \\[{name2}\\]'
        with pytest.raises(AssertionError, match=msg):
            tm.assert_index_equal(idx1, idx2)
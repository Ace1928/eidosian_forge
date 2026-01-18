from datetime import (
import itertools
import numpy as np
import pytest
from pandas.core.dtypes.cast import construct_1d_object_array_from_listlike
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_constructor_mismatched_codes_levels(idx):
    codes = [np.array([1]), np.array([2]), np.array([3])]
    levels = ['a']
    msg = 'Length of levels and codes must be the same'
    with pytest.raises(ValueError, match=msg):
        MultiIndex(levels=levels, codes=codes)
    length_error = 'On level 0, code max \\(3\\) >= length of level \\(1\\)\\. NOTE: this index is in an inconsistent state'
    label_error = 'Unequal code lengths: \\[4, 2\\]'
    code_value_error = 'On level 0, code value \\(-2\\) < -1'
    with pytest.raises(ValueError, match=length_error):
        MultiIndex(levels=[['a'], ['b']], codes=[[0, 1, 2, 3], [0, 3, 4, 1]])
    with pytest.raises(ValueError, match=label_error):
        MultiIndex(levels=[['a'], ['b']], codes=[[0, 0, 0, 0], [0, 0]])
    with pytest.raises(ValueError, match=length_error):
        idx.copy().set_levels([['a'], ['b']])
    with pytest.raises(ValueError, match=label_error):
        idx.copy().set_codes([[0, 0, 0, 0], [0, 0]])
    idx.copy().set_codes(codes=[[0, 0, 0, 0], [0, 0]], verify_integrity=False)
    with pytest.raises(ValueError, match=code_value_error):
        MultiIndex(levels=[['a'], ['b']], codes=[[0, -2], [0, 0]])
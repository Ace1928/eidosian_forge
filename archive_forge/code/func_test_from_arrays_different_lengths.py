from datetime import (
import itertools
import numpy as np
import pytest
from pandas.core.dtypes.cast import construct_1d_object_array_from_listlike
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('idx1, idx2', [([1, 2, 3], ['a', 'b']), ([], ['a', 'b']), ([1, 2, 3], [])])
def test_from_arrays_different_lengths(idx1, idx2):
    msg = '^all arrays must be same length$'
    with pytest.raises(ValueError, match=msg):
        MultiIndex.from_arrays([idx1, idx2])
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_comparison_with_unknown_scalars(self):
    cat = Categorical([1, 2, 3], ordered=True)
    msg = 'Invalid comparison between dtype=category and int'
    with pytest.raises(TypeError, match=msg):
        cat < 4
    with pytest.raises(TypeError, match=msg):
        cat > 4
    with pytest.raises(TypeError, match=msg):
        4 < cat
    with pytest.raises(TypeError, match=msg):
        4 > cat
    tm.assert_numpy_array_equal(cat == 4, np.array([False, False, False]))
    tm.assert_numpy_array_equal(cat != 4, np.array([True, True, True]))
from datetime import (
import itertools
import numpy as np
import pytest
from pandas.core.dtypes.cast import construct_1d_object_array_from_listlike
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('non_frame', [Series([1, 2, 3, 4]), [1, 2, 3, 4], [[1, 2], [3, 4], [5, 6]], Index([1, 2, 3, 4]), np.array([[1, 2], [3, 4], [5, 6]]), 27])
def test_from_frame_error(non_frame):
    with pytest.raises(TypeError, match='Input must be a DataFrame'):
        MultiIndex.from_frame(non_frame)
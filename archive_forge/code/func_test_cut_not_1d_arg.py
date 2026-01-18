import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.types import CategoricalDtype
import pandas.core.reshape.tile as tmod
@pytest.mark.parametrize('arg', [2, np.eye(2), DataFrame(np.eye(2))])
@pytest.mark.parametrize('cut_func', [cut, qcut])
def test_cut_not_1d_arg(arg, cut_func):
    msg = 'Input array must be 1 dimensional'
    with pytest.raises(ValueError, match=msg):
        cut_func(arg, 2)
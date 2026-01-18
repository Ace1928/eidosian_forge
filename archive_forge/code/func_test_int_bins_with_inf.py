import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.types import CategoricalDtype
import pandas.core.reshape.tile as tmod
@pytest.mark.parametrize('data', [[0, 1, 2, 3, 4, np.inf], [-np.inf, 0, 1, 2, 3, 4], [-np.inf, 0, 1, 2, 3, 4, np.inf]])
def test_int_bins_with_inf(data):
    msg = 'cannot specify integer `bins` when input data contains infinity'
    with pytest.raises(ValueError, match=msg):
        cut(data, bins=3)
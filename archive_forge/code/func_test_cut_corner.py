import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.types import CategoricalDtype
import pandas.core.reshape.tile as tmod
@pytest.mark.parametrize('x,bins,msg', [([], 2, 'Cannot cut empty array'), ([1, 2, 3], 0.5, '`bins` should be a positive integer')])
def test_cut_corner(x, bins, msg):
    with pytest.raises(ValueError, match=msg):
        cut(x, bins)
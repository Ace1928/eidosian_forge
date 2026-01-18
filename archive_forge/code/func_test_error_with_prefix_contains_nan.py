import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_error_with_prefix_contains_nan(dummies_basic):
    dummies_basic['col2_c'] = dummies_basic['col2_c'].astype('float64')
    dummies_basic.loc[2, 'col2_c'] = np.nan
    with pytest.raises(ValueError, match="Dummy DataFrame contains NA value in column: 'col2_c'"):
        from_dummies(dummies_basic, sep='_')
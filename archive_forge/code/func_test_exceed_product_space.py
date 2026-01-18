import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.core.reshape.util import cartesian_product
def test_exceed_product_space(self):
    msg = 'Product space too large to allocate arrays!'
    with pytest.raises(ValueError, match=msg):
        dims = [np.arange(0, 22, dtype=np.int16) for i in range(12)] + [np.arange(15128, dtype=np.int16)]
        cartesian_product(X=dims)
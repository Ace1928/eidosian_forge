from datetime import (
import pickle
import numpy as np
import pytest
from pandas._libs.missing import NA
from pandas.core.dtypes.common import is_scalar
import pandas as pd
import pandas._testing as tm
def test_logical_and():
    assert NA & True is NA
    assert True & NA is NA
    assert NA & False is False
    assert False & NA is False
    assert NA & NA is NA
    msg = 'unsupported operand type'
    with pytest.raises(TypeError, match=msg):
        NA & 5
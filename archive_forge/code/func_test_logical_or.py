from datetime import (
import pickle
import numpy as np
import pytest
from pandas._libs.missing import NA
from pandas.core.dtypes.common import is_scalar
import pandas as pd
import pandas._testing as tm
def test_logical_or():
    assert NA | True is True
    assert True | NA is True
    assert NA | False is NA
    assert False | NA is NA
    assert NA | NA is NA
    msg = 'unsupported operand type'
    with pytest.raises(TypeError, match=msg):
        NA | 5
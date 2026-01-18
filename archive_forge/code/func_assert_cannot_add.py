import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
def assert_cannot_add(left, right, msg='cannot add'):
    """
    Helper to assert that left and right cannot be added.

    Parameters
    ----------
    left : object
    right : object
    msg : str, default "cannot add"
    """
    with pytest.raises(TypeError, match=msg):
        left + right
    with pytest.raises(TypeError, match=msg):
        right + left
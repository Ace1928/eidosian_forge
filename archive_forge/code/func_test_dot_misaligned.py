import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_dot_misaligned(self, obj, other):
    msg = 'matrices are not aligned'
    with pytest.raises(ValueError, match=msg):
        obj.dot(other.T)
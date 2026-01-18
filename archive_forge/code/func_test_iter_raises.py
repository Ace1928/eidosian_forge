from datetime import (
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.core.strings.accessor import StringMethods
from pandas.tests.strings import object_pyarrow_numpy
def test_iter_raises():
    ser = Series(['foo', 'bar'])
    with pytest.raises(TypeError, match="'StringMethods' object is not iterable"):
        iter(ser.str)
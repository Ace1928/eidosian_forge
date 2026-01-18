from datetime import (
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.core.strings.accessor import StringMethods
from pandas.tests.strings import object_pyarrow_numpy
def test_cat_on_bytes_raises():
    lhs = Series(np.array(list('abc'), 'S1').astype(object))
    rhs = Series(np.array(list('def'), 'S1').astype(object))
    msg = "Cannot use .str.cat with values of inferred dtype 'bytes'"
    with pytest.raises(TypeError, match=msg):
        lhs.str.cat(rhs)
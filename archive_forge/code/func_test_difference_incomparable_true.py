from datetime import datetime
import operator
import numpy as np
import pytest
from pandas._libs import lib
from pandas.core.dtypes.cast import find_common_type
from pandas import (
import pandas._testing as tm
from pandas.api.types import (
@pytest.mark.parametrize('opname', ['difference', 'symmetric_difference'])
def test_difference_incomparable_true(self, opname):
    a = Index([3, Timestamp('2000'), 1])
    b = Index([2, Timestamp('1999'), 1])
    op = operator.methodcaller(opname, b, sort=True)
    msg = "'<' not supported between instances of 'Timestamp' and 'int'"
    with pytest.raises(TypeError, match=msg):
        op(a)
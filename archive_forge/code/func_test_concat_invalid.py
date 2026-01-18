from io import StringIO
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('obj', [1, {}, [1, 2], (1, 2)])
def test_concat_invalid(self, obj):
    df1 = DataFrame(range(2))
    msg = f"cannot concatenate object of type '{type(obj)}'; only Series and DataFrame objs are valid"
    with pytest.raises(TypeError, match=msg):
        concat([df1, obj])
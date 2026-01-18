from datetime import datetime
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('args', ['foo', datetime(2000, 1, 1, 0, 0)])
def test_constructor_invalid_args_wrong_type(self, args):
    msg = f'Wrong type {type(args)} for value {args}'
    with pytest.raises(TypeError, match=msg):
        RangeIndex(args)
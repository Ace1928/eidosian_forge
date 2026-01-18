from itertools import chain
import re
import numpy as np
import pytest
from pandas.errors import SpecificationError
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('how, args', [('pct_change', ()), ('nsmallest', (1, ['a', 'b'])), ('tail', 1)])
def test_apply_str_axis_1_raises(how, args):
    df = DataFrame({'a': [1, 2], 'b': [3, 4]})
    msg = f'Operation {how} does not support axis=1'
    with pytest.raises(ValueError, match=msg):
        df.apply(how, axis=1, args=args)
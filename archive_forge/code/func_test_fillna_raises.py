import collections
import numpy as np
import pytest
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('fillna_kwargs, msg', [({'value': 1, 'method': 'ffill'}, "Cannot specify both 'value' and 'method'."), ({}, "Must specify a fill 'value' or 'method'."), ({'method': 'bad'}, 'Invalid fill method. Expecting .* bad'), ({'value': Series([1, 2, 3, 4, 'a'])}, 'Cannot setitem on a Categorical with a new category')])
def test_fillna_raises(self, fillna_kwargs, msg):
    cat = Categorical([1, 2, 3, None, None])
    if len(fillna_kwargs) == 1 and 'value' in fillna_kwargs:
        err = TypeError
    else:
        err = ValueError
    with pytest.raises(err, match=msg):
        cat.fillna(**fillna_kwargs)
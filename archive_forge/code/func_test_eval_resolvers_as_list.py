import operator
import numpy as np
import pytest
from pandas.errors import (
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.computation.check import NUMEXPR_INSTALLED
def test_eval_resolvers_as_list(self):
    df = DataFrame(np.random.default_rng(2).standard_normal((10, 2)), columns=list('ab'))
    dict1 = {'a': 1}
    dict2 = {'b': 2}
    assert df.eval('a + b', resolvers=[dict1, dict2]) == dict1['a'] + dict2['b']
    assert pd.eval('a + b', resolvers=[dict1, dict2]) == dict1['a'] + dict2['b']
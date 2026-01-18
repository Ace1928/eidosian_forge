import datetime
import functools
from functools import partial
import re
import numpy as np
import pytest
from pandas.errors import SpecificationError
from pandas.core.dtypes.common import is_integer_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.groupby.grouper import Grouping
@pytest.mark.parametrize('inp', [pd.NamedAgg(column='anything', aggfunc='min'), ('anything', 'min'), ['anything', 'min']])
def test_named_agg_nametuple(self, inp):
    s = Series([1, 1, 2, 2, 3, 3, 4, 5])
    msg = f'func is expected but received {type(inp).__name__}'
    with pytest.raises(TypeError, match=msg):
        s.groupby(s.values).agg(a=inp)
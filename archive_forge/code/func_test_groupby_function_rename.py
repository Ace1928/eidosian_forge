from string import ascii_lowercase
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.core.groupby.base import (
def test_groupby_function_rename(mframe):
    grp = mframe.groupby(level='second')
    for name in ['sum', 'prod', 'min', 'max', 'first', 'last']:
        f = getattr(grp, name)
        assert f.__name__ == name
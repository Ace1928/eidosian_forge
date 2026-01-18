import builtins
from io import StringIO
import numpy as np
import pytest
from pandas._libs import lib
from pandas.errors import UnsupportedFunctionCall
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import nanops
from pandas.tests.groupby import get_groupby_method_args
from pandas.util import _test_decorators as td
def test_groupby_mean_no_overflow():
    df = DataFrame({'user': ['A', 'A', 'A', 'A', 'A'], 'connections': [4970, 4749, 4719, 4704, 18446744073699999744]})
    assert df.groupby('user')['connections'].mean()['A'] == 3689348814740003840
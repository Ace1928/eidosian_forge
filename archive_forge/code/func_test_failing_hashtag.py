import operator
import numpy as np
import pytest
from pandas.errors import (
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.computation.check import NUMEXPR_INSTALLED
def test_failing_hashtag(self, df):
    msg = 'Failed to parse backticks'
    with pytest.raises(SyntaxError, match=msg):
        df.query('`foo#bar` > 4')
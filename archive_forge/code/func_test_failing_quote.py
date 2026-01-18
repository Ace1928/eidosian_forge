import operator
import numpy as np
import pytest
from pandas.errors import (
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.computation.check import NUMEXPR_INSTALLED
def test_failing_quote(self, df):
    msg = '(Could not convert ).*( to a valid Python identifier.)'
    with pytest.raises(SyntaxError, match=msg):
        df.query("`it's` > `that's`")
from datetime import datetime
import re
import numpy as np
import pytest
from pandas._libs import lib
from pandas.errors import UnsupportedFunctionCall
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.indexes.datetimes import date_range
def test_agg_specificationerror_invalid_names(cases):
    msg = "Column\\(s\\) \\['B'\\] do not exist"
    with pytest.raises(KeyError, match=msg):
        cases[['A']].agg({'A': ['sum', 'std'], 'B': ['mean', 'std']})
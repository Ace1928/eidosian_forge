import datetime
from decimal import Decimal
import operator
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import ops
def test_add_string(self):
    index = pd.Index(['a', 'b', 'c'])
    index2 = index + 'foo'
    assert 'a' not in index2
    assert 'afoo' in index2
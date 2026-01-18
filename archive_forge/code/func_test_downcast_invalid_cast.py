import decimal
import numpy as np
from numpy import iinfo
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_downcast_invalid_cast():
    data = ['1', 2, 3]
    invalid_downcast = 'unsigned-integer'
    msg = 'invalid downcasting method provided'
    with pytest.raises(ValueError, match=msg):
        to_numeric(data, downcast=invalid_downcast)
from collections import (
from decimal import Decimal
import math
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_map_decimal(string_series):
    result = string_series.map(lambda x: Decimal(str(x)))
    assert result.dtype == np.object_
    assert isinstance(result.iloc[0], Decimal)
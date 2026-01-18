from collections import (
from decimal import Decimal
import math
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_map_float_to_string_precision():
    ser = Series(1 / 3)
    result = ser.map(lambda val: str(val)).to_dict()
    expected = {0: '0.3333333333333333'}
    assert result == expected
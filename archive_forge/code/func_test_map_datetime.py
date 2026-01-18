from collections import (
from decimal import Decimal
import math
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_map_datetime(datetime_series):
    result = datetime_series.map(lambda x: x * 2)
    tm.assert_series_equal(result, datetime_series * 2)
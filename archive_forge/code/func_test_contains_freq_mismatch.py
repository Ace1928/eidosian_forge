from datetime import datetime
import re
import numpy as np
import pytest
from pandas._libs.tslibs import period as libperiod
from pandas.errors import InvalidIndexError
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_contains_freq_mismatch(self):
    rng = period_range('2007-01', freq='M', periods=10)
    assert Period('2007-01', freq='M') in rng
    assert Period('2007-01', freq='D') not in rng
    assert Period('2007-01', freq='2M') not in rng
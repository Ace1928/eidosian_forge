from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_format_pre_1900_dates(self):
    rng = date_range('1/1/1850', '1/1/1950', freq='A-DEC')
    rng.format()
    ts = Series(1, index=rng)
    repr(ts)
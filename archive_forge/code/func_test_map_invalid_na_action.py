from datetime import datetime
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tseries.offsets import BDay
def test_map_invalid_na_action(float_frame):
    with pytest.raises(ValueError, match="na_action must be .*Got 'abc'"):
        float_frame.map(lambda x: len(str(x)), na_action='abc')
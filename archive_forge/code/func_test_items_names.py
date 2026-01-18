import datetime
import numpy as np
import pytest
from pandas.compat import (
from pandas import (
import pandas._testing as tm
def test_items_names(self, float_string_frame):
    for k, v in float_string_frame.items():
        assert v.name == k
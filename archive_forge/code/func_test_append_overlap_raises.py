import datetime as dt
from itertools import combinations
import dateutil
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_append_overlap_raises(self, float_frame):
    msg = 'Indexes have overlapping values'
    with pytest.raises(ValueError, match=msg):
        float_frame._append(float_frame, verify_integrity=True)
from datetime import timedelta
from decimal import Decimal
import re
from dateutil.tz import tzlocal
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.compat import (
from pandas.compat.numpy import np_version_gt2
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import (
def test_idxmin_axis_2(self, float_frame):
    frame = float_frame
    msg = 'No axis named 2 for object type DataFrame'
    with pytest.raises(ValueError, match=msg):
        frame.idxmin(axis=2)
from datetime import datetime
import dateutil
import numpy as np
import pytest
import pytz
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_astype_uint(self):
    arr = date_range('2000', periods=2, name='idx')
    with pytest.raises(TypeError, match="Do obj.astype\\('int64'\\)"):
        arr.astype('uint64')
    with pytest.raises(TypeError, match="Do obj.astype\\('int64'\\)"):
        arr.astype('uint32')
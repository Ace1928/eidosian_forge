from datetime import datetime
import dateutil
import numpy as np
import pytest
import pytz
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_astype_tznaive_to_tzaware(self):
    idx = date_range('20170101', periods=4)
    idx = idx._with_freq(None)
    msg = 'Cannot use .astype to convert from timezone-naive'
    with pytest.raises(TypeError, match=msg):
        idx.astype('datetime64[ns, US/Eastern]')
    with pytest.raises(TypeError, match=msg):
        idx._data.astype('datetime64[ns, US/Eastern]')
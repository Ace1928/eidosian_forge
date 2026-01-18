import datetime
import dateutil
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_how_thresh_param_incompatible(self):
    df = DataFrame([1, 2, pd.NA])
    msg = 'You cannot set both the how and thresh arguments at the same time'
    with pytest.raises(TypeError, match=msg):
        df.dropna(how='all', thresh=2)
    with pytest.raises(TypeError, match=msg):
        df.dropna(how='any', thresh=2)
    with pytest.raises(TypeError, match=msg):
        df.dropna(how=None, thresh=None)
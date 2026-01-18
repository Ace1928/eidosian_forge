from datetime import datetime
import dateutil.tz
import numpy as np
import pytest
import pytz
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_dti_repr_short(self):
    dr = pd.date_range(start='1/1/2012', periods=1)
    repr(dr)
    dr = pd.date_range(start='1/1/2012', periods=2)
    repr(dr)
    dr = pd.date_range(start='1/1/2012', periods=3)
    repr(dr)
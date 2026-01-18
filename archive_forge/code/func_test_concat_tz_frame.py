import datetime as dt
from datetime import datetime
import dateutil
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_concat_tz_frame(self):
    df2 = DataFrame({'A': Timestamp('20130102', tz='US/Eastern'), 'B': Timestamp('20130603', tz='CET')}, index=range(5))
    df3 = concat([df2.A.to_frame(), df2.B.to_frame()], axis=1)
    tm.assert_frame_equal(df2, df3)
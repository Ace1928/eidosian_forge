import datetime as dt
from datetime import datetime
import dateutil
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_concat_datetime_datetime64_frame(self):
    rows = []
    rows.append([datetime(2010, 1, 1), 1])
    rows.append([datetime(2010, 1, 2), 'hi'])
    df2_obj = DataFrame.from_records(rows, columns=['date', 'test'])
    ind = date_range(start='2000/1/1', freq='D', periods=10)
    df1 = DataFrame({'date': ind, 'test': range(10)})
    concat([df1, df2_obj])
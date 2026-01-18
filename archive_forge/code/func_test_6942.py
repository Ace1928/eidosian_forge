import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_6942(indexer_al):
    start = Timestamp('2014-04-01')
    t1 = Timestamp('2014-04-23 12:42:38.883082')
    t2 = Timestamp('2014-04-24 01:33:30.040039')
    dti = date_range(start, periods=1)
    orig = DataFrame(index=dti, columns=['timenow', 'Live'])
    df = orig.copy()
    indexer_al(df)[start, 'timenow'] = t1
    df['Live'] = True
    df.at[start, 'timenow'] = t2
    assert df.iloc[0, 0] == t2
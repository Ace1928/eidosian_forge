from datetime import (
import re
import numpy as np
import pytest
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape.concat import concat
from pandas.core.reshape.merge import (
def test_join_append_timedeltas2(self):
    td = np.timedelta64(300000000)
    lhs = DataFrame(Series([td, td], index=['A', 'B']))
    rhs = DataFrame(Series([td], index=['A']))
    result = lhs.join(rhs, rsuffix='r', how='left')
    expected = DataFrame({'0': Series([td, td], index=list('AB')), '0r': Series([td, pd.NaT], index=list('AB'))})
    tm.assert_frame_equal(result, expected)
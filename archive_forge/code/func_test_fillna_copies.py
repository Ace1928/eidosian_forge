import numpy as np
import pytest
from pandas._libs.tslibs import iNaT
from pandas._libs.tslibs.period import IncompatibleFrequency
from pandas.core.dtypes.base import _registry as registry
from pandas.core.dtypes.dtypes import PeriodDtype
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import PeriodArray
def test_fillna_copies():
    arr = PeriodArray._from_sequence(['2000', '2001', '2002'], dtype='period[D]')
    result = arr.fillna(pd.Period('2000', 'D'))
    assert result is not arr
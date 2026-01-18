import numpy as np
import pytest
from pandas._libs.tslibs.period import IncompatibleFrequency
from pandas.core.dtypes.dtypes import PeriodDtype
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import PeriodArray
def test_constructor_invalid_quarters(self):
    depr_msg = 'Constructing PeriodIndex from fields is deprecated'
    msg = 'Quarter must be 1 <= q <= 4'
    with pytest.raises(ValueError, match=msg):
        with tm.assert_produces_warning(FutureWarning, match=depr_msg):
            PeriodIndex(year=range(2000, 2004), quarter=list(range(4)), freq='Q-DEC')
import numpy as np
import pytest
from pandas._libs import iNaT
from pandas.core.dtypes.dtypes import DatetimeTZDtype
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import DatetimeArray
def test_from_sequence_invalid_type(self):
    mi = pd.MultiIndex.from_product([np.arange(5), np.arange(5)])
    with pytest.raises(TypeError, match='Cannot create a DatetimeArray'):
        DatetimeArray._from_sequence(mi, dtype='M8[ns]')
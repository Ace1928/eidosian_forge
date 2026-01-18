from datetime import timedelta
import numpy as np
import pytest
import pandas as pd
from pandas import Timedelta
import pandas._testing as tm
from pandas.core.arrays import (
def test_as_unit_raises(self, tda):
    with pytest.raises(ValueError, match='Supported units'):
        tda.as_unit('D')
    tdi = pd.Index(tda)
    with pytest.raises(ValueError, match='Supported units'):
        tdi.as_unit('D')
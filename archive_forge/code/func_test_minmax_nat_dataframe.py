from datetime import (
from decimal import Decimal
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import nanops
from pandas.core.arrays.string_arrow import ArrowStringArrayNumpySemantics
@pytest.mark.parametrize('nat_df', [DataFrame([NaT, NaT]), DataFrame([NaT, Timedelta('nat')]), DataFrame([Timedelta('nat'), Timedelta('nat')])])
def test_minmax_nat_dataframe(self, nat_df):
    assert nat_df.min()[0] is NaT
    assert nat_df.max()[0] is NaT
    assert nat_df.min(skipna=False)[0] is NaT
    assert nat_df.max(skipna=False)[0] is NaT
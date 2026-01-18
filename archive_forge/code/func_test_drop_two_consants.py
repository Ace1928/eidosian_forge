from statsmodels.compat.pandas import (
from statsmodels.compat.pytest import pytest_warns
from collections.abc import Hashable
import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.deterministic import (
def test_drop_two_consants(time_index):
    tt = TimeTrend(constant=True, order=1)
    dp = DeterministicProcess(time_index, constant=True, additional_terms=[tt], drop=True)
    assert dp.in_sample().shape[1] == 2
    dp2 = DeterministicProcess(time_index, additional_terms=[tt], drop=True)
    pd.testing.assert_frame_equal(dp.in_sample(), dp2.in_sample())
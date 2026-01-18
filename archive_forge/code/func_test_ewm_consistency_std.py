import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('bias', [True, False])
def test_ewm_consistency_std(all_data, adjust, ignore_na, min_periods, bias):
    com = 3.0
    var_x = all_data.ewm(com=com, min_periods=min_periods, adjust=adjust, ignore_na=ignore_na).var(bias=bias)
    assert not (var_x < 0).any().any()
    std_x = all_data.ewm(com=com, min_periods=min_periods, adjust=adjust, ignore_na=ignore_na).std(bias=bias)
    assert not (std_x < 0).any().any()
    tm.assert_equal(var_x, std_x * std_x)
    cov_x_x = all_data.ewm(com=com, min_periods=min_periods, adjust=adjust, ignore_na=ignore_na).cov(all_data, bias=bias)
    assert not (cov_x_x < 0).any().any()
    tm.assert_equal(var_x, cov_x_x)
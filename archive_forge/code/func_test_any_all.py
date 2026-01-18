import numpy as np
import pytest
import pandas as pd
@pytest.mark.parametrize('values, exp_any, exp_all, exp_any_noskip, exp_all_noskip', [([True, pd.NA], True, True, True, pd.NA), ([False, pd.NA], False, False, pd.NA, False), ([pd.NA], False, True, pd.NA, pd.NA), ([], False, True, False, True), ([True, True], True, True, True, True), ([False, False], False, False, False, False)])
def test_any_all(values, exp_any, exp_all, exp_any_noskip, exp_all_noskip):
    exp_any = pd.NA if exp_any is pd.NA else np.bool_(exp_any)
    exp_all = pd.NA if exp_all is pd.NA else np.bool_(exp_all)
    exp_any_noskip = pd.NA if exp_any_noskip is pd.NA else np.bool_(exp_any_noskip)
    exp_all_noskip = pd.NA if exp_all_noskip is pd.NA else np.bool_(exp_all_noskip)
    for con in [pd.array, pd.Series]:
        a = con(values, dtype='boolean')
        assert a.any() is exp_any
        assert a.all() is exp_all
        assert a.any(skipna=False) is exp_any_noskip
        assert a.all(skipna=False) is exp_all_noskip
        assert np.any(a.any()) is exp_any
        assert np.all(a.all()) is exp_all
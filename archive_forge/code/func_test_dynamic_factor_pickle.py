import pickle
import os
import tempfile
import pytest
from statsmodels import datasets
from statsmodels.tsa.statespace import (sarimax, structural, varmax,
from numpy.testing import assert_allclose
def test_dynamic_factor_pickle(temp_filename):
    mod = varmax.VARMAX(macrodata[['realgdp', 'realcons']].diff().iloc[1:].values, order=(1, 0))
    pkl_mod = pickle.loads(pickle.dumps(mod))
    res = mod.smooth(mod.start_params)
    pkl_res = pkl_mod.smooth(mod.start_params)
    assert_allclose(res.params, pkl_res.params)
    assert_allclose(res.bse, pkl_res.bse)
    assert_allclose(res.llf, pkl_res.llf)
    res.summary()
    res.save(temp_filename)
    res2 = varmax.VARMAXResults.load(temp_filename)
    assert_allclose(res.params, res2.params)
    assert_allclose(res.bse, res2.bse)
    assert_allclose(res.llf, res2.llf)
import os
import sys
from packaging.version import Version, parse
import numpy as np
from numpy.testing import assert_allclose, assert_
import pandas as pd
def check_ttest_tvalues(results):
    res = results
    mat = np.eye(len(res.params))
    tt = res.t_test(mat)
    assert_allclose(tt.effect, res.params, rtol=1e-12)
    assert_allclose(np.squeeze(tt.sd), res.bse, rtol=1e-10)
    assert_allclose(np.squeeze(tt.tvalue), res.tvalues, rtol=1e-12)
    assert_allclose(tt.pvalue, res.pvalues, rtol=5e-10)
    assert_allclose(tt.conf_int(), res.conf_int(), rtol=1e-10)
    table_res = np.column_stack((res.params, res.bse, res.tvalues, res.pvalues, res.conf_int()))
    table2 = tt.summary_frame().values
    assert_allclose(table2, table_res, rtol=1e-12)
    assert_(hasattr(res, 'use_t'))
    tt = res.t_test(mat[0])
    tt.summary()
    pvalues = np.asarray(res.pvalues)
    assert_allclose(tt.pvalue, pvalues[0], rtol=5e-10)
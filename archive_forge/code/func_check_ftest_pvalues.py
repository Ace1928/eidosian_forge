import os
import sys
from packaging.version import Version, parse
import numpy as np
from numpy.testing import assert_allclose, assert_
import pandas as pd
def check_ftest_pvalues(results):
    """
    Check that the outputs of `res.wald_test` produces pvalues that
    match res.pvalues.

    Check that the string representations of `res.summary()` and (possibly)
    `res.summary2()` correctly label either the t or z-statistic.

    Parameters
    ----------
    results : Results

    Raises
    ------
    AssertionError
    """
    res = results
    use_t = res.use_t
    k_vars = len(res.params)
    pvals = [res.wald_test(np.eye(k_vars)[k], use_f=use_t, scalar=True).pvalue for k in range(k_vars)]
    assert_allclose(pvals, res.pvalues, rtol=5e-10, atol=1e-25)
    pvals = [res.wald_test(np.eye(k_vars)[k], scalar=True).pvalue for k in range(k_vars)]
    assert_allclose(pvals, res.pvalues, rtol=5e-10, atol=1e-25)
    string_use_t = 'P>|z|' if use_t is False else 'P>|t|'
    summ = str(res.summary())
    assert_(string_use_t in summ)
    try:
        summ2 = str(res.summary2())
    except AttributeError:
        pass
    else:
        assert_(string_use_t in summ2)
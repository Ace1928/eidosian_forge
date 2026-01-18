import numpy as np
from numpy.testing import (
import pytest
import statsmodels.datasets.interest_inflation.data as e6
from statsmodels.tools.testing import assert_equal
from statsmodels.tsa.vector_ar.tests.JMulTi_results.parse_jmulti_vecm_output import (
from statsmodels.tsa.vector_ar.util import seasonal_dummies
from statsmodels.tsa.vector_ar.var_model import VARProcess
from statsmodels.tsa.vector_ar.vecm import (
def test_ml_alpha():
    if debug_mode:
        if 'alpha' not in to_test:
            return
        print('\n\nALPHA', end='')
    for ds in datasets:
        for dt in ds.dt_s_list:
            if debug_mode:
                print('\n' + dt_s_tup_to_string(dt) + ': ', end='')
            exog = results_sm_exog[ds][dt].exog is not None
            exog_coint = results_sm_exog_coint[ds][dt].exog_coint is not None
            err_msg = build_err_msg(ds, dt, 'alpha')
            obtained = results_sm[ds][dt].alpha
            obtained_exog = results_sm_exog[ds][dt].alpha
            obtained_exog_coint = results_sm_exog_coint[ds][dt].alpha
            desired = results_ref[ds][dt]['est']['alpha']
            assert_allclose(obtained, desired, rtol, atol, False, err_msg)
            if exog:
                assert_equal(obtained_exog, obtained, 'WITH EXOG: ' + err_msg)
            if exog_coint:
                assert_equal(obtained_exog_coint, obtained, 'WITH EXOG_COINT: ' + err_msg)
            if debug_mode and dont_test_se_t_p:
                continue
            obt = results_sm[ds][dt].stderr_alpha
            obt_exog = results_sm_exog[ds][dt].stderr_alpha
            obt_exog_coint = results_sm_exog_coint[ds][dt].stderr_alpha
            des = results_ref[ds][dt]['se']['alpha']
            assert_allclose(obt, des, rtol, atol, False, 'STANDARD ERRORS\n' + err_msg)
            if exog:
                assert_equal(obt_exog, obt, 'WITH EXOG: ' + err_msg)
            if exog_coint:
                assert_equal(obt_exog_coint, obt, 'WITH EXOG_COINT: ' + err_msg)
            obt = results_sm[ds][dt].tvalues_alpha
            obt_exog = results_sm_exog[ds][dt].tvalues_alpha
            obt_exog_coint = results_sm_exog_coint[ds][dt].tvalues_alpha
            des = results_ref[ds][dt]['t']['alpha']
            assert_allclose(obt, des, rtol, atol, False, 't-VALUES\n' + err_msg)
            if exog:
                assert_equal(obt_exog, obt, 'WITH EXOG: ' + err_msg)
            if exog_coint:
                assert_equal(obt_exog_coint, obt, 'WITH EXOG_COINT: ' + err_msg)
            obt = results_sm[ds][dt].pvalues_alpha
            obt_exog = results_sm_exog[ds][dt].pvalues_alpha
            obt_exog_coint = results_sm_exog_coint[ds][dt].pvalues_alpha
            des = results_ref[ds][dt]['p']['alpha']
            assert_allclose(obt, des, rtol, atol, False, 'p-VALUES\n' + err_msg)
            if exog:
                assert_equal(obt_exog, obt, 'WITH EXOG: ' + err_msg)
            if exog_coint:
                assert_equal(obt_exog_coint, obt, 'WITH EXOG_COINT: ' + err_msg)
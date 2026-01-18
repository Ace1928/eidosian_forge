import numpy as np
from numpy.testing import (
import pytest
import statsmodels.datasets.interest_inflation.data as e6
from statsmodels.tools.testing import assert_equal
from statsmodels.tsa.vector_ar.tests.JMulTi_results.parse_jmulti_vecm_output import (
from statsmodels.tsa.vector_ar.util import seasonal_dummies
from statsmodels.tsa.vector_ar.var_model import VARProcess
from statsmodels.tsa.vector_ar.vecm import (
def test_ml_c():
    if debug_mode:
        if 'C' not in to_test:
            return
        print('\n\nDET_COEF', end='')
    for ds in datasets:
        for dt in ds.dt_s_list:
            if debug_mode:
                print('\n' + dt_s_tup_to_string(dt) + ': ', end='')
            exog = results_sm_exog[ds][dt].exog is not None
            exog_coint = results_sm_exog_coint[ds][dt].exog_coint is not None
            C_obt = results_sm[ds][dt].det_coef
            C_obt_exog = results_sm_exog[ds][dt].det_coef
            C_obt_exog_coint = results_sm_exog_coint[ds][dt].det_coef
            se_C_obt = results_sm[ds][dt].stderr_det_coef
            se_C_obt_exog = results_sm_exog[ds][dt].stderr_det_coef
            se_C_obt_exog_coint = results_sm_exog_coint[ds][dt].stderr_det_coef
            t_C_obt = results_sm[ds][dt].tvalues_det_coef
            t_C_obt_exog = results_sm_exog[ds][dt].tvalues_det_coef
            t_C_obt_exog_coint = results_sm_exog_coint[ds][dt].tvalues_det_coef
            p_C_obt = results_sm[ds][dt].pvalues_det_coef
            p_C_obt_exog = results_sm_exog[ds][dt].pvalues_det_coef
            p_C_obt_exog_coint = results_sm_exog_coint[ds][dt].pvalues_det_coef
            if 'C' not in results_ref[ds][dt]['est'].keys():
                if C_obt.size == 0 and se_C_obt.size == 0 and (t_C_obt.size == 0) and (p_C_obt.size == 0):
                    assert_(True)
                    continue
            desired = results_ref[ds][dt]['est']['C']
            dt_string = dt_s_tup_to_string(dt)
            if 'co' in dt_string:
                err_msg = build_err_msg(ds, dt, 'CONST')
                const_obt = C_obt[:, :1]
                const_obt_exog = C_obt_exog[:, :1]
                const_obt_exog_coint = C_obt_exog_coint[:, :1]
                const_des = desired[:, :1]
                C_obt = C_obt[:, 1:]
                C_obt_exog = C_obt_exog[:, 1:]
                C_obt_exog_coint = C_obt_exog_coint[:, 1:]
                desired = desired[:, 1:]
                assert_allclose(const_obt, const_des, rtol, atol, False, err_msg)
                if exog:
                    assert_equal(const_obt_exog, const_obt, 'WITH EXOG: ' + err_msg)
                if exog_coint:
                    assert_equal(const_obt_exog_coint, const_obt, 'WITH EXOG_COINT: ' + err_msg)
            if 's' in dt_string:
                err_msg = build_err_msg(ds, dt, 'SEASONAL')
                if 'lo' in dt_string:
                    seas_obt = C_obt[:, :-1]
                    seas_obt_exog = C_obt_exog[:, :-1]
                    seas_obt_exog_coint = C_obt_exog_coint[:, :-1]
                    seas_des = desired[:, :-1]
                else:
                    seas_obt = C_obt
                    seas_obt_exog = C_obt_exog
                    seas_obt_exog_coint = C_obt_exog_coint
                    seas_des = desired
                assert_allclose(seas_obt, seas_des, rtol, atol, False, err_msg)
                if exog:
                    assert_equal(seas_obt_exog, seas_obt, 'WITH EXOG: ' + err_msg)
                if exog_coint:
                    assert_equal(seas_obt_exog_coint, seas_obt, 'WITH EXOG_COINT: ' + err_msg)
            if 'lo' in dt_string:
                err_msg = build_err_msg(ds, dt, 'LINEAR TREND')
                lt_obt = C_obt[:, -1:]
                lt_obt_exog = C_obt_exog[:, -1:]
                lt_obt_exog_coint = C_obt_exog_coint[:, -1:]
                lt_des = desired[:, -1:]
                assert_allclose(lt_obt, lt_des, rtol, atol, False, err_msg)
                if exog:
                    assert_equal(lt_obt_exog, lt_obt, 'WITH EXOG: ' + err_msg)
                if exog_coint:
                    assert_equal(lt_obt_exog_coint, lt_obt, 'WITH EXOG_COINT: ' + err_msg)
            if debug_mode and dont_test_se_t_p:
                continue
            se_desired = results_ref[ds][dt]['se']['C']
            if 'co' in dt_string:
                err_msg = build_err_msg(ds, dt, 'SE CONST')
                se_const_obt = se_C_obt[:, 0][:, None]
                se_C_obt = se_C_obt[:, 1:]
                se_const_obt_exog = se_C_obt_exog[:, 0][:, None]
                se_C_obt_exog = se_C_obt_exog[:, 1:]
                se_const_obt_exog_coint = se_C_obt_exog_coint[:, 0][:, None]
                se_C_obt_exog_coint = se_C_obt_exog_coint[:, 1:]
                se_const_des = se_desired[:, 0][:, None]
                se_desired = se_desired[:, 1:]
                assert_allclose(se_const_obt, se_const_des, rtol, atol, False, err_msg)
                if exog:
                    assert_equal(se_const_obt_exog, se_const_obt, 'WITH EXOG: ' + err_msg)
                if exog_coint:
                    assert_equal(se_const_obt_exog_coint, se_const_obt, 'WITH EXOG_COINT: ' + err_msg)
            if 's' in dt_string:
                err_msg = build_err_msg(ds, dt, 'SE SEASONAL')
                if 'lo' in dt_string:
                    se_seas_obt = se_C_obt[:, :-1]
                    se_seas_obt_exog = se_C_obt_exog[:, :-1]
                    se_seas_obt_exog_coint = se_C_obt_exog_coint[:, :-1]
                    se_seas_des = se_desired[:, :-1]
                else:
                    se_seas_obt = se_C_obt
                    se_seas_obt_exog = se_C_obt_exog
                    se_seas_obt_exog_coint = se_C_obt_exog_coint
                    se_seas_des = se_desired
                assert_allclose(se_seas_obt, se_seas_des, rtol, atol, False, err_msg)
                if exog:
                    assert_equal(se_seas_obt_exog, se_seas_obt, 'WITH EXOG: ' + err_msg)
                if exog_coint:
                    assert_equal(se_seas_obt_exog_coint, se_seas_obt, 'WITH EXOG_COINT: ' + err_msg)
                if 'lo' in dt_string:
                    err_msg = build_err_msg(ds, dt, 'SE LIN. TREND')
                    se_lt_obt = se_C_obt[:, -1:]
                    se_lt_obt_exog = se_C_obt_exog[:, -1:]
                    se_lt_obt_exog_coint = se_C_obt_exog_coint[:, -1:]
                    se_lt_des = se_desired[:, -1:]
                    assert_allclose(se_lt_obt, se_lt_des, rtol, atol, False, err_msg)
                    if exog:
                        assert_equal(se_lt_obt_exog, se_lt_obt, 'WITH EXOG: ' + err_msg)
                    if exog_coint:
                        assert_equal(se_lt_obt_exog_coint, se_lt_obt, 'WITH EXOG_COINT: ' + err_msg)
            t_desired = results_ref[ds][dt]['t']['C']
            if 'co' in dt_string:
                t_const_obt = t_C_obt[:, 0][:, None]
                t_C_obt = t_C_obt[:, 1:]
                t_const_obt_exog = t_C_obt_exog[:, 0][:, None]
                t_C_obt_exog = t_C_obt_exog[:, 1:]
                t_const_obt_exog_coint = t_C_obt_exog_coint[:, 0][:, None]
                t_C_obt_exog_coint = t_C_obt_exog_coint[:, 1:]
                t_const_des = t_desired[:, 0][:, None]
                t_desired = t_desired[:, 1:]
                assert_allclose(t_const_obt, t_const_des, rtol, atol, False, build_err_msg(ds, dt, 'T CONST'))
                if exog:
                    assert_equal(t_const_obt_exog, t_const_obt, 'WITH EXOG: ' + err_msg)
                if exog_coint:
                    assert_equal(t_const_obt_exog_coint, t_const_obt, 'WITH EXOG_COINT: ' + err_msg)
            if 's' in dt_string:
                if 'lo' in dt_string:
                    t_seas_obt = t_C_obt[:, :-1]
                    t_seas_obt_exog = t_C_obt_exog[:, :-1]
                    t_seas_obt_exog_coint = t_C_obt_exog_coint[:, :-1]
                    t_seas_des = t_desired[:, :-1]
                else:
                    t_seas_obt = t_C_obt
                    t_seas_obt_exog = t_C_obt_exog
                    t_seas_obt_exog_coint = t_C_obt_exog_coint
                    t_seas_des = t_desired
                assert_allclose(t_seas_obt, t_seas_des, rtol, atol, False, build_err_msg(ds, dt, 'T SEASONAL'))
                if exog:
                    assert_equal(t_seas_obt_exog, t_seas_obt, 'WITH EXOG' + err_msg)
                if exog_coint:
                    assert_equal(t_seas_obt_exog_coint, t_seas_obt, 'WITH EXOG_COINT' + err_msg)
            if 'lo' in dt_string:
                t_lt_obt = t_C_obt[:, -1:]
                t_lt_obt_exog = t_C_obt_exog[:, -1:]
                t_lt_obt_exog_coint = t_C_obt_exog_coint[:, -1:]
                t_lt_des = t_desired[:, -1:]
                assert_allclose(t_lt_obt, t_lt_des, rtol, atol, False, build_err_msg(ds, dt, 'T LIN. TREND'))
                if exog:
                    assert_equal(t_lt_obt_exog, t_lt_obt, 'WITH EXOG' + err_msg)
                if exog_coint:
                    assert_equal(t_lt_obt_exog_coint, t_lt_obt, 'WITH EXOG_COINT' + err_msg)
            p_desired = results_ref[ds][dt]['p']['C']
            if 'co' in dt_string:
                p_const_obt = p_C_obt[:, 0][:, None]
                p_C_obt = p_C_obt[:, 1:]
                p_const_obt_exog = p_C_obt_exog[:, 0][:, None]
                p_C_obt_exog = p_C_obt_exog[:, 1:]
                p_const_obt_exog_coint = p_C_obt_exog_coint[:, 0][:, None]
                p_C_obt_exo_cointg = p_C_obt_exog_coint[:, 1:]
                p_const_des = p_desired[:, 0][:, None]
                p_desired = p_desired[:, 1:]
                assert_allclose(p_const_obt, p_const_des, rtol, atol, False, build_err_msg(ds, dt, 'P CONST'))
                if exog:
                    assert_equal(p_const_obt, p_const_obt_exog, 'WITH EXOG' + err_msg)
                if exog_coint:
                    assert_equal(p_const_obt, p_const_obt_exog_coint, 'WITH EXOG_COINT' + err_msg)
            if 's' in dt_string:
                if 'lo' in dt_string:
                    p_seas_obt = p_C_obt[:, :-1]
                    p_seas_obt_exog = p_C_obt_exog[:, :-1]
                    p_seas_obt_exog_coint = p_C_obt_exog_coint[:, :-1]
                    p_seas_des = p_desired[:, :-1]
                else:
                    p_seas_obt = p_C_obt
                    p_seas_obt_exog = p_C_obt_exog
                    p_seas_obt_exog_coint = p_C_obt_exog_coint
                    p_seas_des = p_desired
                assert_allclose(p_seas_obt, p_seas_des, rtol, atol, False, build_err_msg(ds, dt, 'P SEASONAL'))
                if exog:
                    assert_equal(p_seas_obt_exog, p_seas_obt, 'WITH EXOG' + err_msg)
                if exog_coint:
                    assert_equal(p_seas_obt_exog_coint, p_seas_obt, 'WITH EXOG_COINT' + err_msg)
            if 'lo' in dt_string:
                p_lt_obt = p_C_obt[:, -1:]
                p_lt_obt_exog = p_C_obt_exog[:, -1:]
                p_lt_obt_exog_coint = p_C_obt_exog_coint[:, -1:]
                p_lt_des = p_desired[:, -1:]
                assert_allclose(p_lt_obt, p_lt_des, rtol, atol, False, build_err_msg(ds, dt, 'P LIN. TREND'))
                if exog:
                    assert_equal(p_lt_obt_exog, p_lt_obt, 'WITH EXOG' + err_msg)
                if exog_coint:
                    assert_equal(p_lt_obt_exog_coint, p_lt_obt, 'WITH EXOG_COINT' + err_msg)
import numpy as np
from numpy.testing import (
import pytest
import statsmodels.datasets.interest_inflation.data as e6
from statsmodels.tools.testing import assert_equal
from statsmodels.tsa.vector_ar.tests.JMulTi_results.parse_jmulti_vecm_output import (
from statsmodels.tsa.vector_ar.util import seasonal_dummies
from statsmodels.tsa.vector_ar.var_model import VARProcess
from statsmodels.tsa.vector_ar.vecm import (
def test_lag_order_selection():
    if debug_mode:
        if 'lag order' not in to_test:
            return
        else:
            print('\n\nLAG ORDER SELECTION', end='')
    for ds in datasets:
        for dt in ds.dt_s_list:
            if debug_mode:
                print('\n' + dt_s_tup_to_string(dt) + ': ', end='')
            deterministic = dt[0]
            endog_tot = data[ds]
            trend = 'n' if dt[0] == 'nc' else dt[0]
            obtained_all = select_order(endog_tot, 10, deterministic=dt[0], seasons=dt[1])
            deterministic_outside_exog = ''
            if 'co' in deterministic:
                deterministic_outside_exog += 'co'
            if 'lo' in deterministic and dt[1] == 0:
                deterministic_outside_exog += 'lo'
            exog_model = results_sm_exog[ds][dt].exog
            exog = exog_model is not None
            exog_coint_model = results_sm_exog_coint[ds][dt].exog_coint
            exog_coint = exog_coint_model is not None
            obtained_all_exog = select_order(endog_tot, 10, deterministic_outside_exog, seasons=0, exog=exog_model)
            obtained_all_exog_coint = select_order(endog_tot, 10, 'n', seasons=dt[1], exog_coint=exog_coint_model)
            for ic in ['aic', 'fpe', 'hqic', 'bic']:
                err_msg = build_err_msg(ds, dt, 'LAG ORDER SELECTION - ' + ic.upper())
                obtained = getattr(obtained_all, ic)
                desired = results_ref[ds][dt]['lagorder'][ic]
                assert_allclose(obtained, desired, rtol, atol, False, err_msg)
                if exog:
                    assert_equal(getattr(obtained_all_exog, ic), getattr(obtained_all, ic), 'WITH EXOG' + err_msg)
                if exog_coint:
                    assert_equal(getattr(obtained_all_exog_coint, ic), getattr(obtained_all, ic), 'WITH EXOG_COINT' + err_msg)
            obtained_all.summary()
            str(obtained_all)
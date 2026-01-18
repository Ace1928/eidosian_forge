import numpy as np
import pytest
from statsmodels.tsa.statespace import (
from statsmodels.tsa.statespace.tests.test_impulse_responses import TVSS
from numpy.testing import assert_allclose
def check_filter_output(mod, periods, atol=0):
    if isinstance(mod, mlemodel.MLEModel):
        res_mv = mod.ssm.filter()
        mod.ssm.filter()
        kfilter = mod.ssm._kalman_filter
        kfilter.seek(0, True)
        kfilter.univariate_filter[periods] = 1
        for _ in range(mod.nobs):
            next(kfilter)
        res_switch = mod.ssm.results_class(mod.ssm)
        res_switch.update_representation(mod.ssm)
        res_switch.update_filter(kfilter)
        mod.ssm.filter_univariate = True
        res_uv = mod.ssm.filter()
    else:
        res_mv, res_switch, res_uv = mod
    assert_allclose(res_switch.llf, res_mv.llf)
    assert_allclose(res_switch.llf, res_uv.llf)
    assert_allclose(res_switch.scale, res_mv.scale)
    assert_allclose(res_switch.scale, res_uv.scale)
    attrs = ['forecasts_error_diffuse_cov', 'predicted_state', 'predicted_state_cov', 'predicted_diffuse_state_cov', 'filtered_state', 'filtered_state_cov', 'llf_obs']
    for attr in attrs:
        attr_mv = getattr(res_mv, attr)
        attr_uv = getattr(res_uv, attr)
        attr_switch = getattr(res_switch, attr)
        if attr_mv is None:
            continue
        assert_allclose(attr_switch, attr_mv, atol=atol)
        assert_allclose(attr_switch, attr_uv, atol=atol)
    attrs = ['forecasts_error', 'forecasts_error_cov', 'kalman_gain']
    for attr in attrs:
        actual = getattr(res_switch, attr).copy()
        desired = getattr(res_mv, attr).copy()
        actual[..., periods] = 0
        desired[..., periods] = 0
        assert_allclose(actual, desired, atol=atol)
        actual = getattr(res_switch, attr)[..., periods]
        desired = getattr(res_uv, attr)[..., periods]
        assert_allclose(actual, desired, atol=atol)
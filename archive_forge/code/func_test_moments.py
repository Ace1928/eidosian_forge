import sys
import numpy as np
import numpy.testing as npt
import pytest
from pytest import raises as assert_raises
from scipy.integrate import IntegrationWarning
import itertools
from scipy import stats
from .common_tests import (check_normalization, check_moment,
from scipy.stats._distr_params import distcont
from scipy.stats._distn_infrastructure import rv_continuous_frozen
@pytest.mark.slow
@pytest.mark.parametrize('distname,arg,normalization_ok,higher_ok,moment_ok,is_xfailing', cases_test_moments())
def test_moments(distname, arg, normalization_ok, higher_ok, moment_ok, is_xfailing):
    try:
        distfn = getattr(stats, distname)
    except TypeError:
        distfn = distname
        distname = 'rv_histogram_instance'
    with npt.suppress_warnings() as sup:
        sup.filter(IntegrationWarning, 'The integral is probably divergent, or slowly convergent.')
        sup.filter(IntegrationWarning, 'The maximum number of subdivisions.')
        sup.filter(IntegrationWarning, 'The algorithm does not converge.')
        if is_xfailing:
            sup.filter(IntegrationWarning)
        m, v, s, k = distfn.stats(*arg, moments='mvsk')
        with np.errstate(all='ignore'):
            if normalization_ok:
                check_normalization(distfn, arg, distname)
            if higher_ok:
                check_mean_expect(distfn, arg, m, distname)
                check_skew_expect(distfn, arg, m, v, s, distname)
                check_var_expect(distfn, arg, m, v, distname)
                check_kurt_expect(distfn, arg, m, v, k, distname)
                check_munp_expect(distfn, arg, distname)
        check_loc_scale(distfn, arg, m, v, distname)
        if moment_ok:
            check_moment(distfn, arg, m, v, distname)
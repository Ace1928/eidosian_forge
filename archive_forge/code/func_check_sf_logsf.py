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
def check_sf_logsf(distfn, args, msg):
    points = np.array([0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0])
    vals = distfn.ppf(points, *args)
    vals = vals[np.isfinite(vals)]
    sf = distfn.sf(vals, *args)
    logsf = distfn.logsf(vals, *args)
    sf = sf[sf != 0]
    logsf = logsf[np.isfinite(logsf)]
    msg += ' - logsf-log(sf) relationship'
    npt.assert_almost_equal(np.log(sf), logsf, decimal=7, err_msg=msg)
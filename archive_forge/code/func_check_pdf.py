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
def check_pdf(distfn, arg, msg):
    median = distfn.ppf(0.5, *arg)
    eps = 1e-06
    pdfv = distfn.pdf(median, *arg)
    if pdfv < 0.0001 or pdfv > 10000.0:
        median = median + 0.1
        pdfv = distfn.pdf(median, *arg)
    cdfdiff = (distfn.cdf(median + eps, *arg) - distfn.cdf(median - eps, *arg)) / eps / 2.0
    msg += ' - cdf-pdf relationship'
    npt.assert_almost_equal(pdfv, cdfdiff, decimal=DECIMAL, err_msg=msg)
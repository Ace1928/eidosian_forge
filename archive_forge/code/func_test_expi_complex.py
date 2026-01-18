import numpy as np
from numpy.testing import assert_, assert_allclose
from numpy import pi
import pytest
import itertools
from scipy._lib import _pep440
import scipy.special as sc
from scipy.special._testutils import (
from scipy.special._mptestutils import (
from scipy.special._ufuncs import (
@check_version(mpmath, '0.10')
def test_expi_complex():
    dataset = []
    for r in np.logspace(-99, 2, 10):
        for p in np.linspace(0, 2 * np.pi, 30):
            z = r * np.exp(1j * p)
            dataset.append((z, complex(mpmath.ei(z))))
    dataset = np.array(dataset, dtype=np.cdouble)
    FuncData(sc.expi, dataset, 0, 1).check()
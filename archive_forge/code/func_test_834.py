import pytest
import numpy as np
from numpy.testing import assert_allclose
import scipy.special as sc
def test_834(self):
    a = sc.exp1(-complex(19.999999))
    b = sc.exp1(-complex(19.9999991))
    assert_allclose(a.imag, b.imag, atol=0, rtol=1e-15)
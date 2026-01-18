import pytest
import numpy as np
from numpy.testing import assert_allclose
import scipy.special as sc
def test_continuity_on_positive_real_axis(self):
    assert_allclose(sc.expi(complex(1, 0)), sc.expi(complex(1, -0.0)), atol=0, rtol=1e-15)
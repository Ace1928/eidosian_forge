from mpmath import *
from mpmath.libmp import *
import random
def test_fractional_pow():
    mp.dps = 15
    assert mpf(16) ** 2.5 == 1024
    assert mpf(64) ** 0.5 == 8
    assert mpf(64) ** (-0.5) == 0.125
    assert mpf(16) ** (-2.5) == 0.0009765625
    assert (mpf(10) ** 0.5).ae(3.162277660168379)
    assert (mpf(10) ** 2.5).ae(316.2277660168379)
    assert (mpf(10) ** (-0.5)).ae(0.31622776601683794)
    assert (mpf(10) ** (-2.5)).ae(0.0031622776601683794)
    assert (mpf(10) ** 0.3).ae(1.9952623149688795)
    assert (mpf(10) ** (-0.3)).ae(0.5011872336272722)
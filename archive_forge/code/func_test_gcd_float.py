import pytest
from monty.fractions import gcd, gcd_float, lcm
def test_gcd_float():
    vs = [6.2, 12.4, 15.5 + 5e-09]
    assert gcd_float(vs, 1e-08) == pytest.approx(3.1)
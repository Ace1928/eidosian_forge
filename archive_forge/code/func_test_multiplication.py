from sympy.sets.ordinals import Ordinal, OmegaPower, ord0, omega
from sympy.testing.pytest import raises
def test_multiplication():
    w = omega
    assert w * (w + 1) == w * w + w
    assert (w + 1) * (w + 1) == w * w + w + 1
    assert w * 1 == w
    assert 1 * w == w
    assert w * ord0 == ord0
    assert ord0 * w == ord0
    assert w ** w == w * w ** w
    assert w ** w * w * w == w ** (w + 2)
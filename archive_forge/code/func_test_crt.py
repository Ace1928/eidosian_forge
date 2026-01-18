from sympy.ntheory.modular import crt, crt1, crt2, solve_congruence
from sympy.testing.pytest import raises
def test_crt():

    def mcrt(m, v, r, symmetric=False):
        assert crt(m, v, symmetric)[0] == r
        mm, e, s = crt1(m)
        assert crt2(m, v, mm, e, s, symmetric) == (r, mm)
    mcrt([2, 3, 5], [0, 0, 0], 0)
    mcrt([2, 3, 5], [1, 1, 1], 1)
    mcrt([2, 3, 5], [-1, -1, -1], -1, True)
    mcrt([2, 3, 5], [-1, -1, -1], 2 * 3 * 5 - 1, False)
    assert crt([656, 350], [811, 133], symmetric=True) == (-56917, 114800)
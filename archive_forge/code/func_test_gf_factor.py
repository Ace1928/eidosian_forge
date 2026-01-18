from sympy.polys.galoistools import (
from sympy.polys.polyerrors import (
from sympy.polys import polyconfig as config
from sympy.polys.domains import ZZ
from sympy.core.numbers import pi
from sympy.ntheory.generate import nextprime
from sympy.testing.pytest import raises
def test_gf_factor():
    assert gf_factor([], 11, ZZ) == (0, [])
    assert gf_factor([1], 11, ZZ) == (1, [])
    assert gf_factor([1, 1], 11, ZZ) == (1, [([1, 1], 1)])
    assert gf_factor_sqf([], 11, ZZ) == (0, [])
    assert gf_factor_sqf([1], 11, ZZ) == (1, [])
    assert gf_factor_sqf([1, 1], 11, ZZ) == (1, [[1, 1]])
    config.setup('GF_FACTOR_METHOD', 'berlekamp')
    assert gf_factor_sqf([], 11, ZZ) == (0, [])
    assert gf_factor_sqf([1], 11, ZZ) == (1, [])
    assert gf_factor_sqf([1, 1], 11, ZZ) == (1, [[1, 1]])
    config.setup('GF_FACTOR_METHOD', 'zassenhaus')
    assert gf_factor_sqf([], 11, ZZ) == (0, [])
    assert gf_factor_sqf([1], 11, ZZ) == (1, [])
    assert gf_factor_sqf([1, 1], 11, ZZ) == (1, [[1, 1]])
    config.setup('GF_FACTOR_METHOD', 'shoup')
    assert gf_factor_sqf(ZZ.map([]), 11, ZZ) == (0, [])
    assert gf_factor_sqf(ZZ.map([1]), 11, ZZ) == (1, [])
    assert gf_factor_sqf(ZZ.map([1, 1]), 11, ZZ) == (1, [[1, 1]])
    f, p = (ZZ.map([1, 0, 0, 1, 0]), 2)
    g = (1, [([1, 0], 1), ([1, 1], 1), ([1, 1, 1], 1)])
    config.setup('GF_FACTOR_METHOD', 'berlekamp')
    assert gf_factor(f, p, ZZ) == g
    config.setup('GF_FACTOR_METHOD', 'zassenhaus')
    assert gf_factor(f, p, ZZ) == g
    config.setup('GF_FACTOR_METHOD', 'shoup')
    assert gf_factor(f, p, ZZ) == g
    g = (1, [[1, 0], [1, 1], [1, 1, 1]])
    config.setup('GF_FACTOR_METHOD', 'berlekamp')
    assert gf_factor_sqf(f, p, ZZ) == g
    config.setup('GF_FACTOR_METHOD', 'zassenhaus')
    assert gf_factor_sqf(f, p, ZZ) == g
    config.setup('GF_FACTOR_METHOD', 'shoup')
    assert gf_factor_sqf(f, p, ZZ) == g
    f, p = (gf_from_int_poly([1, -3, 1, -3, -1, -3, 1], 11), 11)
    g = (1, [([1, 1], 1), ([1, 5, 3], 1), ([1, 2, 3, 4], 1)])
    config.setup('GF_FACTOR_METHOD', 'berlekamp')
    assert gf_factor(f, p, ZZ) == g
    config.setup('GF_FACTOR_METHOD', 'zassenhaus')
    assert gf_factor(f, p, ZZ) == g
    config.setup('GF_FACTOR_METHOD', 'shoup')
    assert gf_factor(f, p, ZZ) == g
    f, p = ([1, 5, 8, 4], 11)
    g = (1, [([1, 1], 1), ([1, 2], 2)])
    config.setup('GF_FACTOR_METHOD', 'berlekamp')
    assert gf_factor(f, p, ZZ) == g
    config.setup('GF_FACTOR_METHOD', 'zassenhaus')
    assert gf_factor(f, p, ZZ) == g
    config.setup('GF_FACTOR_METHOD', 'shoup')
    assert gf_factor(f, p, ZZ) == g
    f, p = ([1, 1, 10, 1, 0, 10, 10, 10, 0, 0], 11)
    g = (1, [([1, 0], 2), ([1, 9, 5], 1), ([1, 3, 0, 8, 5, 2], 1)])
    config.setup('GF_FACTOR_METHOD', 'berlekamp')
    assert gf_factor(f, p, ZZ) == g
    config.setup('GF_FACTOR_METHOD', 'zassenhaus')
    assert gf_factor(f, p, ZZ) == g
    config.setup('GF_FACTOR_METHOD', 'shoup')
    assert gf_factor(f, p, ZZ) == g
    f, p = (gf_from_dict({32: 1, 0: 1}, 11, ZZ), 11)
    g = (1, [([1, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 10], 1), ([1, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 10], 1)])
    config.setup('GF_FACTOR_METHOD', 'berlekamp')
    assert gf_factor(f, p, ZZ) == g
    config.setup('GF_FACTOR_METHOD', 'zassenhaus')
    assert gf_factor(f, p, ZZ) == g
    config.setup('GF_FACTOR_METHOD', 'shoup')
    assert gf_factor(f, p, ZZ) == g
    f, p = (gf_from_dict({32: ZZ(8), 0: ZZ(5)}, 11, ZZ), 11)
    g = (8, [([1, 3], 1), ([1, 8], 1), ([1, 0, 9], 1), ([1, 2, 2], 1), ([1, 9, 2], 1), ([1, 0, 5, 0, 7], 1), ([1, 0, 6, 0, 7], 1), ([1, 0, 0, 0, 1, 0, 0, 0, 6], 1), ([1, 0, 0, 0, 10, 0, 0, 0, 6], 1)])
    config.setup('GF_FACTOR_METHOD', 'berlekamp')
    assert gf_factor(f, p, ZZ) == g
    config.setup('GF_FACTOR_METHOD', 'zassenhaus')
    assert gf_factor(f, p, ZZ) == g
    config.setup('GF_FACTOR_METHOD', 'shoup')
    assert gf_factor(f, p, ZZ) == g
    f, p = (gf_from_dict({63: ZZ(8), 0: ZZ(5)}, 11, ZZ), 11)
    g = (8, [([1, 7], 1), ([1, 4, 5], 1), ([1, 6, 8, 2], 1), ([1, 9, 9, 2], 1), ([1, 0, 0, 9, 0, 0, 4], 1), ([1, 2, 0, 8, 4, 6, 4], 1), ([1, 2, 3, 8, 0, 6, 4], 1), ([1, 2, 6, 0, 8, 4, 4], 1), ([1, 3, 3, 1, 6, 8, 4], 1), ([1, 5, 6, 0, 8, 6, 4], 1), ([1, 6, 2, 7, 9, 8, 4], 1), ([1, 10, 4, 7, 10, 7, 4], 1), ([1, 10, 10, 1, 4, 9, 4], 1)])
    config.setup('GF_FACTOR_METHOD', 'berlekamp')
    assert gf_factor(f, p, ZZ) == g
    config.setup('GF_FACTOR_METHOD', 'zassenhaus')
    assert gf_factor(f, p, ZZ) == g
    config.setup('GF_FACTOR_METHOD', 'shoup')
    assert gf_factor(f, p, ZZ) == g
    p = ZZ(nextprime(int((2 ** 15 * pi).evalf())))
    f = gf_from_dict({15: 1, 1: 1, 0: 1}, p, ZZ)
    assert gf_sqf_p(f, p, ZZ) is True
    g = (1, [([1, 22730, 68144], 1), ([1, 81553, 77449, 86810, 4724], 1), ([1, 86276, 56779, 14859, 31575], 1), ([1, 15347, 95022, 84569, 94508, 92335], 1)])
    config.setup('GF_FACTOR_METHOD', 'zassenhaus')
    assert gf_factor(f, p, ZZ) == g
    config.setup('GF_FACTOR_METHOD', 'shoup')
    assert gf_factor(f, p, ZZ) == g
    g = (1, [[1, 22730, 68144], [1, 81553, 77449, 86810, 4724], [1, 86276, 56779, 14859, 31575], [1, 15347, 95022, 84569, 94508, 92335]])
    config.setup('GF_FACTOR_METHOD', 'zassenhaus')
    assert gf_factor_sqf(f, p, ZZ) == g
    config.setup('GF_FACTOR_METHOD', 'shoup')
    assert gf_factor_sqf(f, p, ZZ) == g
    p = ZZ(nextprime(int((2 ** 4 * pi).evalf())))
    f = ZZ.map([1, 2, 5, 26, 41, 39, 38])
    assert gf_sqf_p(f, p, ZZ) is True
    g = (1, [([1, 44, 26], 1), ([1, 11, 25, 18, 30], 1)])
    config.setup('GF_FACTOR_METHOD', 'zassenhaus')
    assert gf_factor(f, p, ZZ) == g
    config.setup('GF_FACTOR_METHOD', 'shoup')
    assert gf_factor(f, p, ZZ) == g
    g = (1, [[1, 44, 26], [1, 11, 25, 18, 30]])
    config.setup('GF_FACTOR_METHOD', 'zassenhaus')
    assert gf_factor_sqf(f, p, ZZ) == g
    config.setup('GF_FACTOR_METHOD', 'shoup')
    assert gf_factor_sqf(f, p, ZZ) == g
    config.setup('GF_FACTOR_METHOD', 'other')
    raises(KeyError, lambda: gf_factor([1, 1], 11, ZZ))
    config.setup('GF_FACTOR_METHOD')
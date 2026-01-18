from sympy.concrete.summations import summation
from sympy.core.containers import Dict
from sympy.core.mul import Mul
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.combinatorial.factorials import factorial as fac
from sympy.core.evalf import bitcount
from sympy.core.numbers import Integer, Rational
from sympy.ntheory import (totient,
from sympy.ntheory.factor_ import (smoothness, smoothness_p, proper_divisors,
from sympy.testing.pytest import raises, slow
from sympy.utilities.iterables import capture
@slow
def test_factorint():
    assert primefactors(123456) == [2, 3, 643]
    assert factorint(0) == {0: 1}
    assert factorint(1) == {}
    assert factorint(-1) == {-1: 1}
    assert factorint(-2) == {-1: 1, 2: 1}
    assert factorint(-16) == {-1: 1, 2: 4}
    assert factorint(2) == {2: 1}
    assert factorint(126) == {2: 1, 3: 2, 7: 1}
    assert factorint(123456) == {2: 6, 3: 1, 643: 1}
    assert factorint(5951757) == {3: 1, 7: 1, 29: 2, 337: 1}
    assert factorint(64015937) == {7993: 1, 8009: 1}
    assert factorint(2 ** 2 ** 6 + 1) == {274177: 1, 67280421310721: 1}
    assert factorint(10 ** 38 - 1) == {3: 2, 11: 1, 909090909090909091: 1, 1111111111111111111: 1}
    assert factorint(28300421052393658575) == {3: 1, 5: 2, 11: 2, 43: 1, 2063: 2, 4127: 1, 4129: 1}
    assert factorint(2063 ** 2 * 4127 ** 1 * 4129 ** 1) == {2063: 2, 4127: 1, 4129: 1}
    assert factorint(2347 ** 2 * 7039 ** 1 * 7043 ** 1) == {2347: 2, 7039: 1, 7043: 1}
    assert factorint(0, multiple=True) == [0]
    assert factorint(1, multiple=True) == []
    assert factorint(-1, multiple=True) == [-1]
    assert factorint(-2, multiple=True) == [-1, 2]
    assert factorint(-16, multiple=True) == [-1, 2, 2, 2, 2]
    assert factorint(2, multiple=True) == [2]
    assert factorint(24, multiple=True) == [2, 2, 2, 3]
    assert factorint(126, multiple=True) == [2, 3, 3, 7]
    assert factorint(123456, multiple=True) == [2, 2, 2, 2, 2, 2, 3, 643]
    assert factorint(5951757, multiple=True) == [3, 7, 29, 29, 337]
    assert factorint(64015937, multiple=True) == [7993, 8009]
    assert factorint(2 ** 2 ** 6 + 1, multiple=True) == [274177, 67280421310721]
    assert factorint(fac(1, evaluate=False)) == {}
    assert factorint(fac(7, evaluate=False)) == {2: 4, 3: 2, 5: 1, 7: 1}
    assert factorint(fac(15, evaluate=False)) == {2: 11, 3: 6, 5: 3, 7: 2, 11: 1, 13: 1}
    assert factorint(fac(20, evaluate=False)) == {2: 18, 3: 8, 5: 4, 7: 2, 11: 1, 13: 1, 17: 1, 19: 1}
    assert factorint(fac(23, evaluate=False)) == {2: 19, 3: 9, 5: 4, 7: 3, 11: 2, 13: 1, 17: 1, 19: 1, 23: 1}
    assert multiproduct(factorint(fac(200))) == fac(200)
    assert multiproduct(factorint(fac(200, evaluate=False))) == fac(200)
    for b, e in factorint(fac(150)).items():
        assert e == fac_multiplicity(150, b)
    for b, e in factorint(fac(150, evaluate=False)).items():
        assert e == fac_multiplicity(150, b)
    assert factorint(103005006059 ** 7) == {103005006059: 7}
    assert factorint(31337 ** 191) == {31337: 191}
    assert factorint(2 ** 1000 * 3 ** 500 * 257 ** 127 * 383 ** 60) == {2: 1000, 3: 500, 257: 127, 383: 60}
    assert len(factorint(fac(10000))) == 1229
    assert len(factorint(fac(10000, evaluate=False))) == 1229
    assert factorint(12932983746293756928584532764589230) == {2: 1, 5: 1, 73: 1, 727719592270351: 1, 63564265087747: 1, 383: 1}
    assert factorint(727719592270351) == {727719592270351: 1}
    assert factorint(2 ** 64 + 1, use_trial=False) == factorint(2 ** 64 + 1)
    for n in range(60000):
        assert multiproduct(factorint(n)) == n
    assert pollard_rho(2 ** 64 + 1, seed=1) == 274177
    assert pollard_rho(19, seed=1) is None
    assert factorint(3, limit=2) == {3: 1}
    assert factorint(12345) == {3: 1, 5: 1, 823: 1}
    assert factorint(12345, limit=3) == {4115: 1, 3: 1}
    assert factorint(1, limit=1) == {}
    assert factorint(0, 3) == {0: 1}
    assert factorint(12, limit=1) == {12: 1}
    assert factorint(30, limit=2) == {2: 1, 15: 1}
    assert factorint(16, limit=2) == {2: 4}
    assert factorint(124, limit=3) == {2: 2, 31: 1}
    assert factorint(4 * 31 ** 2, limit=3) == {2: 2, 31: 2}
    p1 = nextprime(2 ** 32)
    p2 = nextprime(2 ** 16)
    p3 = nextprime(p2)
    assert factorint(p1 * p2 * p3) == {p1: 1, p2: 1, p3: 1}
    assert factorint(13 * 17 * 19, limit=15) == {13: 1, 17 * 19: 1}
    assert factorint(1951 * 15013 * 15053, limit=2000) == {225990689: 1, 1951: 1}
    assert factorint(primorial(17) + 1, use_pm1=0) == {int(19026377261): 1, 3467: 1, 277: 1, 105229: 1}
    a = nextprime(2 ** 2 ** 8)
    b = nextprime(a + 2 ** 2 ** 4)
    assert 'Fermat' in capture(lambda: factorint(a * b, verbose=1))
    raises(ValueError, lambda: pollard_rho(4))
    raises(ValueError, lambda: pollard_pm1(3))
    raises(ValueError, lambda: pollard_pm1(10, B=2))
    n = nextprime(2 ** 16) * nextprime(2 ** 17) * nextprime(1901)
    assert 'with primes' in capture(lambda: factorint(n, verbose=1))
    capture(lambda: factorint(nextprime(2 ** 16) * 1012, verbose=1))
    n = nextprime(2 ** 17)
    capture(lambda: factorint(n ** 3, verbose=1))
    capture(lambda: factorint(2 * n, verbose=1))
    n = nextprime(2 ** 17)
    n *= nextprime(n)
    assert '1000' in capture(lambda: factorint(n, limit=1000, verbose=1))
    n *= nextprime(n)
    assert len(factorint(n)) == 3
    assert len(factorint(n, limit=p1)) == 3
    n *= nextprime(2 * n)
    assert '2001' in capture(lambda: factorint(n, limit=2000, verbose=1))
    assert capture(lambda: factorint(n, limit=4000, verbose=1)).count('Pollard') == 2
    n = nextprime(8069)
    n *= nextprime(2 * n) * nextprime(2 * n, 2)
    capture(lambda: factorint(n, verbose=1))
    p1 = nextprime(2 ** 17)
    p2 = nextprime(2 * p1)
    assert factorint((p1 * p2 ** 2) ** 3) == {p1: 3, p2: 6}
    raises(ValueError, lambda: factorint(4.5))
    sans = '2**10*3**3'
    n = {4: 2, 12: 3}
    assert str(factorint(n)) == sans
    assert str(factorint(Dict(n))) == sans
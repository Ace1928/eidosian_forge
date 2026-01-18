from mpmath.libmp import *
from mpmath import mpf, mp
from random import randint, choice, seed
def test_div_1_3():
    a = fi(1)
    b = fi(3)
    c = fi(-1)
    assert mpf_div(a, b, 7, round_floor) == fb('0.01010101')
    assert mpf_div(a, b, 7, round_ceiling) == fb('0.01010110')
    assert mpf_div(a, b, 7, round_down) == fb('0.01010101')
    assert mpf_div(a, b, 7, round_up) == fb('0.01010110')
    assert mpf_div(a, b, 7, round_nearest) == fb('0.01010101')
    assert mpf_div(c, b, 7, round_floor) == fb('-0.01010110')
    assert mpf_div(c, b, 7, round_ceiling) == fb('-0.01010101')
    assert mpf_div(c, b, 7, round_down) == fb('-0.01010101')
    assert mpf_div(c, b, 7, round_up) == fb('-0.01010110')
    assert mpf_div(c, b, 7, round_nearest) == fb('-0.01010101')
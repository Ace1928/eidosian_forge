from mpmath.libmp import *
from mpmath import mpf, mp
from random import randint, choice, seed
def test_div_300():
    q = fi(1000000)
    a = fi(300499999)
    b = fi(300500000)
    c = fi(300500001)
    assert mpf_div(a, q, 9, round_down) == fi(300)
    assert mpf_div(b, q, 9, round_down) == fi(300)
    assert mpf_div(c, q, 9, round_down) == fi(300)
    assert mpf_div(a, q, 9, round_up) == fi(301)
    assert mpf_div(b, q, 9, round_up) == fi(301)
    assert mpf_div(c, q, 9, round_up) == fi(301)
    assert mpf_div(a, q, 9, round_nearest) == fi(300)
    assert mpf_div(b, q, 9, round_nearest) == fi(300)
    assert mpf_div(c, q, 9, round_nearest) == fi(301)
    a = fi(301499999)
    b = fi(301500000)
    c = fi(301500001)
    assert mpf_div(a, q, 9, round_nearest) == fi(301)
    assert mpf_div(b, q, 9, round_nearest) == fi(302)
    assert mpf_div(c, q, 9, round_nearest) == fi(302)
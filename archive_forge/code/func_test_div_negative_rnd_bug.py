from mpmath.libmp import *
from mpmath import mpf, mp
from random import randint, choice, seed
def test_div_negative_rnd_bug():
    mp.dps = 15
    assert -3 / mpf('0.1531879017645047') == mpf('-19.583791966887116')
    assert mpf('-2.6342475750861301') / mpf('0.35126216427941814') == mpf('-7.4993775104985909')
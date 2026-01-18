import math
import pytest
from mpmath import *
def test_elliptic_integrals():
    mp.dps = 15
    assert ellipk(0).ae(pi / 2)
    assert ellipk(0.5).ae(gamma(0.25) ** 2 / (4 * sqrt(pi)))
    assert ellipk(1) == inf
    assert ellipk(1 + 0j) == inf
    assert ellipk(-1).ae('1.3110287771460599052')
    assert ellipk(-2).ae('1.1714200841467698589')
    assert isinstance(ellipk(-2), mpf)
    assert isinstance(ellipe(-2), mpf)
    assert ellipk(-50).ae('0.47103424540873331679')
    mp.dps = 30
    n1 = +fraction(99999, 100000)
    n2 = +fraction(100001, 100000)
    mp.dps = 15
    assert ellipk(n1).ae('7.1427724505817781901')
    assert ellipk(n2).ae(mpc('7.1427417367963090109', '-1.5707923998261688019'))
    assert ellipe(n1).ae('1.0000332138990829170')
    v = ellipe(n2)
    assert v.real.ae('0.999966786328145474069137')
    assert (v.imag * 10 ** 6).ae('7.853952181727432')
    assert ellipk(2).ae(mpc('1.3110287771460599052', '-1.3110287771460599052'))
    assert ellipk(50).ae(mpc('0.22326753950210985451', '-0.47434723226254522087'))
    assert ellipk(3 + 4j).ae(mpc('0.91119556380496500866', '0.63133428324134524388'))
    assert ellipk(3 - 4j).ae(mpc('0.91119556380496500866', '-0.63133428324134524388'))
    assert ellipk(-3 + 4j).ae(mpc('0.95357894880405122483', '0.23093044503746114444'))
    assert ellipk(-3 - 4j).ae(mpc('0.95357894880405122483', '-0.23093044503746114444'))
    assert isnan(ellipk(nan))
    assert isnan(ellipe(nan))
    assert ellipk(inf) == 0
    assert isinstance(ellipk(inf), mpc)
    assert ellipk(-inf) == 0
    assert ellipk(1 + 0j) == inf
    assert ellipe(0).ae(pi / 2)
    assert ellipe(0.5).ae(pi ** (mpf(3) / 2) / gamma(0.25) ** 2 + gamma(0.25) ** 2 / (8 * sqrt(pi)))
    assert ellipe(1) == 1
    assert ellipe(1 + 0j) == 1
    assert ellipe(inf) == mpc(0, inf)
    assert ellipe(-inf) == inf
    assert ellipe(3 + 4j).ae(1.499553520933347 - 1.5778790079127583j)
    assert ellipe(3 - 4j).ae(1.499553520933347 + 1.5778790079127583j)
    assert ellipe(-3 + 4j).ae(2.5804237855343377 - 0.8306096791000414j)
    assert ellipe(-3 - 4j).ae(2.5804237855343377 + 0.8306096791000414j)
    assert ellipe(2).ae(0.5990701173677961 + 0.5990701173677961j)
    assert ellipe('1e-1000000000').ae(pi / 2)
    assert ellipk('1e-1000000000').ae(pi / 2)
    assert ellipe(-pi).ae(2.4535865983838923)
    mp.dps = 50
    assert ellipk(1 / pi).ae('1.724756270009501831744438120951614673874904182624739673')
    assert ellipe(1 / pi).ae('1.437129808135123030101542922290970050337425479058225712')
    assert ellipk(-10 * pi).ae('0.5519067523886233967683646782286965823151896970015484512')
    assert ellipe(-10 * pi).ae('5.926192483740483797854383268707108012328213431657645509')
    v = ellipk(pi)
    assert v.real.ae('0.973089521698042334840454592642137667227167622330325225')
    assert v.imag.ae('-1.156151296372835303836814390793087600271609993858798016')
    v = ellipe(pi)
    assert v.real.ae('0.4632848917264710404078033487934663562998345622611263332')
    assert v.imag.ae('1.0637961621753130852473300451583414489944099504180510966')
    mp.dps = 15
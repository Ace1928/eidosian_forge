import math
import pytest
from mpmath import *
def test_exp_integrals():
    mp.dps = 15
    x = +e
    z = e + sqrt(3) * j
    assert ei(x).ae(8.211681655383616)
    assert li(x).ae(1.8951178163559368)
    assert si(x).ae(1.8210402691475671)
    assert ci(x).ae(0.2139580013403798)
    assert shi(x).ae(4.115207062478462)
    assert chi(x).ae(4.096474592905154)
    assert fresnels(x).ae(0.43718971814978763)
    assert fresnelc(x).ae(0.40177775959024303)
    assert airyai(x).ae(0.010850240156858667)
    assert airybi(x).ae(8.982457485854686)
    assert ei(z).ae(3.7259796949131494 + 7.342132123142244j)
    assert li(z).ae(2.286626581125625 + 1.5042722529726937j)
    assert si(z).ae(2.4812202923766904 + 0.12684703275254833j)
    assert ci(z).ae(0.16925559026945663 - 0.8920207514207803j)
    assert shi(z).ae(1.8581036655934446 + 3.664358429149203j)
    assert chi(z).ae(1.8678760293197048 + 3.6777736939930414j)
    assert fresnels(z / 3).ae(0.03453439719700818 + 0.7548598441882187j)
    assert fresnelc(z / 3).ae(1.2615816459900273 + 0.41794919877506187j)
    assert airyai(z).ae(-0.016255257983905605 - 0.0018045715700210556j)
    assert airybi(z).ae(-4.988561132828834 + 2.0855853787218064j)
    assert li(0) == 0.0
    assert li(1) == -inf
    assert li(inf) == inf
    assert isinstance(li(0.7), mpf)
    assert si(inf).ae(pi / 2)
    assert si(-inf).ae(-pi / 2)
    assert ci(inf) == 0
    assert ci(0) == -inf
    assert isinstance(ei(-0.7), mpf)
    assert airyai(inf) == 0
    assert airybi(inf) == inf
    assert airyai(-inf) == 0
    assert airybi(-inf) == 0
    assert fresnels(inf) == 0.5
    assert fresnelc(inf) == 0.5
    assert fresnels(-inf) == -0.5
    assert fresnelc(-inf) == -0.5
    assert shi(0) == 0
    assert shi(inf) == inf
    assert shi(-inf) == -inf
    assert chi(0) == -inf
    assert chi(inf) == inf
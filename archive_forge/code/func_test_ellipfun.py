import mpmath
import random
import pytest
from mpmath import *
def test_ellipfun():
    mp.dps = 15
    assert ellipfun('ss', 0, 0) == 1
    assert ellipfun('cc', 0, 0) == 1
    assert ellipfun('dd', 0, 0) == 1
    assert ellipfun('nn', 0, 0) == 1
    assert ellipfun('sn', 0.25, 0).ae(sin(0.25))
    assert ellipfun('cn', 0.25, 0).ae(cos(0.25))
    assert ellipfun('dn', 0.25, 0).ae(1)
    assert ellipfun('ns', 0.25, 0).ae(csc(0.25))
    assert ellipfun('nc', 0.25, 0).ae(sec(0.25))
    assert ellipfun('nd', 0.25, 0).ae(1)
    assert ellipfun('sc', 0.25, 0).ae(tan(0.25))
    assert ellipfun('sd', 0.25, 0).ae(sin(0.25))
    assert ellipfun('cd', 0.25, 0).ae(cos(0.25))
    assert ellipfun('cs', 0.25, 0).ae(cot(0.25))
    assert ellipfun('dc', 0.25, 0).ae(sec(0.25))
    assert ellipfun('ds', 0.25, 0).ae(csc(0.25))
    assert ellipfun('sn', 0.25, 1).ae(tanh(0.25))
    assert ellipfun('cn', 0.25, 1).ae(sech(0.25))
    assert ellipfun('dn', 0.25, 1).ae(sech(0.25))
    assert ellipfun('ns', 0.25, 1).ae(coth(0.25))
    assert ellipfun('nc', 0.25, 1).ae(cosh(0.25))
    assert ellipfun('nd', 0.25, 1).ae(cosh(0.25))
    assert ellipfun('sc', 0.25, 1).ae(sinh(0.25))
    assert ellipfun('sd', 0.25, 1).ae(sinh(0.25))
    assert ellipfun('cd', 0.25, 1).ae(1)
    assert ellipfun('cs', 0.25, 1).ae(csch(0.25))
    assert ellipfun('dc', 0.25, 1).ae(1)
    assert ellipfun('ds', 0.25, 1).ae(csch(0.25))
    assert ellipfun('sn', 0.25, 0.5).ae(0.24615967096986147)
    assert ellipfun('cn', 0.25, 0.5).ae(0.9692292898937844)
    assert ellipfun('dn', 0.25, 0.5).ae(0.9847348415659948)
    assert ellipfun('ns', 0.25, 0.5).ae(4.062403870057313)
    assert ellipfun('nc', 0.25, 0.5).ae(1.0317476065024693)
    assert ellipfun('nd', 0.25, 0.5).ae(1.0155017958029489)
    assert ellipfun('sc', 0.25, 0.5).ae(0.25397465134058994)
    assert ellipfun('sd', 0.25, 0.5).ae(0.24997558792415733)
    assert ellipfun('cd', 0.25, 0.5).ae(0.984254084431955)
    assert ellipfun('cs', 0.25, 0.5).ae(3.937400818237411)
    assert ellipfun('dc', 0.25, 0.5).ae(1.0159978158253034)
    assert ellipfun('ds', 0.25, 0.5).ae(4.0003906313579725)
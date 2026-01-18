import collections
import pytest
from ..util.testing import requires
from ..chemistry import Substance, Reaction, Equilibrium, Species
@requires('numpy')
def test_Equilibria_arithmetics():
    es1 = _get_es1()
    e, = es1.rxns
    e2 = 2 * e
    sum2 = e + e
    assert sum2 == e2
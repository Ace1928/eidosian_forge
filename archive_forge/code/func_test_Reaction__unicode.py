from functools import reduce
from operator import attrgetter, add
import sys
from sympy import nsimplify
import pytest
from ..util.arithmeticdict import ArithmeticDict
from ..util.testing import requires
from ..util.parsing import parsing_library
from ..units import default_units, units_library, to_unitless, allclose
from ..chemistry import (
@requires(parsing_library)
def test_Reaction__unicode():
    keys = u'H2O H2 O2'.split()
    subst = {k: Substance.from_formula(k) for k in keys}
    r2 = Reaction.from_string('2 H2O -> 2 H2 + O2', subst)
    assert r2.unicode(subst) == u'2 H₂O → 2 H₂ + O₂'
    r3 = Reaction.from_string("2 H2O -> 2 H2 + O2; 42; name='split'", subst)
    assert r3.unicode(subst) == u'2 H₂O → 2 H₂ + O₂'
    assert r3.unicode(subst, with_name=True) == u'2 H₂O → 2 H₂ + O₂; split'
    assert r3.unicode(subst, with_name=True, with_param=True) == u'2 H₂O → 2 H₂ + O₂; 42; split'
    assert r3.unicode(subst, with_param=True) == u'2 H₂O → 2 H₂ + O₂; 42'
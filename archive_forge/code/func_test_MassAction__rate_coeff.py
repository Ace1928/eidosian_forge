import math
import pytest
from chempy import Reaction
from chempy.units import (
from chempy.util._expr import Log10, Constant
from chempy.util.testing import requires
from ..rates import MassAction, Radiolytic
from .._rates import (
@requires(units_library)
def test_MassAction__rate_coeff():
    perMolar_perSecond = u.perMolar_perSecond
    p1 = MassAction(Constant(perMolar_perSecond) * 10 ** RTPoly([1, 2 * u.kelvin, 3 * u.kelvin ** 2]))
    rcoeff1 = p1.rate_coeff({'temperature': 283.15 * u.K})
    ref1 = 10 ** (1 + 2 / 283.15 + 3 / 283.15 ** 2) * perMolar_perSecond
    assert allclose(rcoeff1, ref1)
    rxn1 = Reaction({'A', 'B'}, {'C'}, p1)
    rat1 = rxn1.rate({'A': 2, 'B': 3, 'temperature': 283.15 * u.K})
    assert allclose(rat1['A'], -2 * 3 * ref1)
    assert allclose(rat1['B'], -2 * 3 * ref1)
    assert allclose(rat1['C'], 2 * 3 * ref1)
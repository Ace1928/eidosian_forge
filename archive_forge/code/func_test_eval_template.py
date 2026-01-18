import math
from chempy import Reaction
from chempy.units import allclose, default_units as u
from ..testing import requires
from ..rendering import eval_template
from ..parsing import get_parsing_context
from chempy.units import units_library
@requires(units_library)
def test_eval_template():
    rendered = eval_template('${2*pi*arg*m**2}', arg=1 / math.pi)
    val = eval(rendered, get_parsing_context())
    assert allclose(val, 2 * u.m ** 2)
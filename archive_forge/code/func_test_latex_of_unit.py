from collections import defaultdict
import pytest
from ..util.testing import requires
from ..units import (
@requires(units_library)
def test_latex_of_unit():
    assert latex_of_unit(u.gram / u.metre ** 2) == '\\mathrm{\\frac{g}{m^{2}}}'
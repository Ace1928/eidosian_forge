from collections import defaultdict
import pytest
from ..util.testing import requires
from ..units import (
@requires(units_library)
def test__sum():
    assert (_sum([0.1 * u.metre, 1 * u.decimetre]) - 2 * u.decimetre) / u.metre == 0
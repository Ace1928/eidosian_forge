from collections import defaultdict
import pytest
from ..util.testing import requires
from ..units import (
@requires(units_library)
def test_joule_html():
    joule_htm = 'kg&sdot;m<sup>2</sup>/s<sup>2</sup>'
    joule = u.J.dimensionality.simplified
    assert joule.html == joule_htm
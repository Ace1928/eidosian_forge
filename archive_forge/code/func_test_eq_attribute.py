from __future__ import annotations
from datetime import (
import pytest
from pandas._libs.tslibs import (
from pandas._libs.tslibs.offsets import (
from pandas import (
from pandas.tests.tseries.offsets.common import assert_offset_equal
@pytest.mark.parametrize('offset_name', ['offset1', 'offset2', 'offset3', 'offset4', 'offset8', 'offset9', 'offset10'])
def test_eq_attribute(self, offset_name, request):
    offset = request.getfixturevalue(offset_name)
    assert offset == offset
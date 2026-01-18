from __future__ import (absolute_import, division, print_function)
import pytest
from .test_core import (
@pytest.mark.skipif(levmar is None, reason='levmar package unavailable')
def test_levmar_fail():
    _test_fail('levmar')
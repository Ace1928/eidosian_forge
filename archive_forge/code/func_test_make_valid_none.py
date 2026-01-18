import numpy as np
import pytest
import shapely
from shapely import (
from shapely.testing import assert_geometries_equal
from shapely.tests.common import (
@pytest.mark.skipif(shapely.geos_version < (3, 8, 0), reason='GEOS < 3.8')
def test_make_valid_none():
    actual = shapely.make_valid(None)
    assert actual is None
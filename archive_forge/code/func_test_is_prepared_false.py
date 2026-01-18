from functools import partial
import numpy as np
import pytest
import shapely
from shapely import LinearRing, LineString, Point
from shapely.tests.common import (
@pytest.mark.parametrize('geometry', all_types + (empty, None))
def test_is_prepared_false(geometry):
    assert not shapely.is_prepared(geometry)
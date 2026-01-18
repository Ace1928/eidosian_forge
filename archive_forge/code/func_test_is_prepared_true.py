from functools import partial
import numpy as np
import pytest
import shapely
from shapely import LinearRing, LineString, Point
from shapely.tests.common import (
@pytest.mark.parametrize('geometry', all_types + (empty,))
def test_is_prepared_true(geometry):
    assert shapely.is_prepared(_prepare_with_copy(geometry))
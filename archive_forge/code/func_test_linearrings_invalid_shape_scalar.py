import numpy as np
import pytest
import shapely
from shapely import (
from shapely.testing import assert_geometries_equal
from shapely.tests.common import (
def test_linearrings_invalid_shape_scalar():
    with pytest.raises(ValueError):
        shapely.linearrings((1, 1))
import platform
import weakref
import numpy as np
import pytest
import shapely
from shapely import (
from shapely.errors import ShapelyDeprecationWarning
from shapely.testing import assert_geometries_equal
@pytest.mark.skipif(shapely.geos_version < (3, 9, 0), reason='GEOS < 3.9')
@pytest.mark.parametrize('op', ['union', 'intersection', 'difference', 'symmetric_difference'])
@pytest.mark.parametrize('grid_size', [0, 1, 2])
def test_binary_op_grid_size(op, grid_size):
    geom1 = shapely.box(0, 0, 2.5, 2.5)
    geom2 = shapely.box(2, 2, 3, 3)
    result = getattr(geom1, op)(geom2, grid_size=grid_size)
    expected = getattr(shapely, op)(geom1, geom2, grid_size=grid_size)
    assert result == expected
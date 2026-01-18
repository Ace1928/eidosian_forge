import platform
import weakref
import numpy as np
import pytest
import shapely
from shapely import (
from shapely.errors import ShapelyDeprecationWarning
from shapely.testing import assert_geometries_equal
def test_linestring():
    assert bool(LineString()) is False
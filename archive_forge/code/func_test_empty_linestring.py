import math
import numpy as np
import pytest
from shapely import (
from shapely.geometry import mapping, shape
from shapely.geometry.base import BaseGeometry, EmptyGeometry
def test_empty_linestring(self):
    assert LineString().is_empty
    assert LineString(None).is_empty
    assert LineString([]).is_empty
    assert LineString(empty_generator()).is_empty
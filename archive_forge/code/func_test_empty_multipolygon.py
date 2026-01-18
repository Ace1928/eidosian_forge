import math
import numpy as np
import pytest
from shapely import (
from shapely.geometry import mapping, shape
from shapely.geometry.base import BaseGeometry, EmptyGeometry
def test_empty_multipolygon(self):
    assert MultiPolygon([]).is_empty
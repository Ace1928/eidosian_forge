import numpy as np
import pytest
import shapely
from shapely import (
from shapely.testing import assert_geometries_equal
from shapely.tests.common import (
def box_tpl(x1, y1, x2, y2):
    return ((x2, y1), (x2, y2), (x1, y2), (x1, y1), (x2, y1))
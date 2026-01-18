import numpy as np
import pytest
import shapely
from shapely import (
from shapely.testing import assert_geometries_equal
from shapely.tests.common import (
def test_clip_by_rect_missing():
    actual = shapely.clip_by_rect(None, 0, 0, 1, 1)
    assert actual is None
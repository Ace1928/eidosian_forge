import numpy as np
import pytest
import shapely
from shapely import LinearRing, LineString, Point
from shapely.coords import CoordinateSequence
@pytest.mark.filterwarnings('ignore:Creating an ndarray from ragged nested sequences:')
def test_from_invalid_dim():
    with pytest.raises(shapely.GEOSException):
        LineString([(1, 2)])
    with pytest.raises((ValueError, TypeError)):
        LineString([(1, 2, 3), (4, 5)])
    with pytest.raises((ValueError, TypeError)):
        LineString([(1, 2), (3, 4, 5)])
    msg = 'The ordinate \\(last\\) dimension should be 2 or 3, got {}'
    with pytest.raises(ValueError, match=msg.format(4)):
        LineString([(1, 2, 3, 4), (4, 5, 6, 7)])
    with pytest.raises(ValueError, match=msg.format(1)):
        LineString([(1,), (4,)])
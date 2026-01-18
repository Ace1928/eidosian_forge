import unittest
import pytest
import shapely
from shapely.geometry import Point, Polygon
def test_binary_predicate_exceptions(self):
    p1 = [(339, 346), (459, 346), (399, 311), (340, 277), (399, 173), (280, 242), (339, 415), (280, 381), (460, 207), (339, 346)]
    p2 = [(339, 207), (280, 311), (460, 138), (399, 242), (459, 277), (459, 415), (399, 381), (519, 311), (520, 242), (519, 173), (399, 450), (339, 207)]
    with pytest.raises(shapely.GEOSException):
        Polygon(p1).within(Polygon(p2))
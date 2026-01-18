import unittest
import pytest
from shapely.algorithms.polylabel import Cell, polylabel
from shapely.errors import TopologicalError
from shapely.geometry import LineString, Point, Polygon
def test_invalid_polygon(self):
    """
        Makes sure that the polylabel function throws an exception when provided
        an invalid polygon.

        """
    bowtie_polygon = Polygon([(0, 0), (0, 20), (10, 10), (20, 20), (20, 0), (10, 10), (0, 0)])
    with pytest.raises(TopologicalError):
        polylabel(bowtie_polygon)
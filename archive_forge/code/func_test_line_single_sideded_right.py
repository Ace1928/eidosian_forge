import unittest
import pytest
from shapely import geometry
from shapely.constructive import BufferCapStyle, BufferJoinStyle
from shapely.geometry.base import CAP_STYLE, JOIN_STYLE
def test_line_single_sideded_right(self):
    g = geometry.LineString([[0, 0], [0, 1]])
    h = g.buffer(-1, quad_segs=1, single_sided=True)
    assert h.geom_type == 'Polygon'
    expected_coord = [(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0), (0.0, 0.0)]
    for index, coord in enumerate(h.exterior.coords):
        assert coord[0] == pytest.approx(expected_coord[index][0])
        assert coord[1] == pytest.approx(expected_coord[index][1])
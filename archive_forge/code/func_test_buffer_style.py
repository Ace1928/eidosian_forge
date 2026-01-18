import unittest
import pytest
from shapely import geometry
from shapely.constructive import BufferCapStyle, BufferJoinStyle
from shapely.geometry.base import CAP_STYLE, JOIN_STYLE
def test_buffer_style(self):
    g = geometry.LineString([[0, 0], [1, 0]])
    h = g.buffer(1, join_style=BufferJoinStyle.round)
    assert h == g.buffer(1, join_style=JOIN_STYLE.round)
    assert h == g.buffer(1, join_style='round')
    h = g.buffer(1, join_style=BufferJoinStyle.mitre)
    assert h == g.buffer(1, join_style=JOIN_STYLE.mitre)
    assert h == g.buffer(1, join_style='mitre')
    h = g.buffer(1, join_style=BufferJoinStyle.bevel)
    assert h == g.buffer(1, join_style=JOIN_STYLE.bevel)
    assert h == g.buffer(1, join_style='bevel')
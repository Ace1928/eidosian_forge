import unittest
import pytest
from shapely import geometry
from shapely.constructive import BufferCapStyle, BufferJoinStyle
from shapely.geometry.base import CAP_STYLE, JOIN_STYLE
def test_cap_style(self):
    g = geometry.LineString([[0, 0], [1, 0]])
    h = g.buffer(1, cap_style=BufferCapStyle.round)
    assert h == g.buffer(1, cap_style=CAP_STYLE.round)
    assert h == g.buffer(1, cap_style='round')
    h = g.buffer(1, cap_style=BufferCapStyle.flat)
    assert h == g.buffer(1, cap_style=CAP_STYLE.flat)
    assert h == g.buffer(1, cap_style='flat')
    h = g.buffer(1, cap_style=BufferCapStyle.square)
    assert h == g.buffer(1, cap_style=CAP_STYLE.square)
    assert h == g.buffer(1, cap_style='square')
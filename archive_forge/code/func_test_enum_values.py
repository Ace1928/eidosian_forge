import unittest
import pytest
from shapely import geometry
from shapely.constructive import BufferCapStyle, BufferJoinStyle
from shapely.geometry.base import CAP_STYLE, JOIN_STYLE
def test_enum_values(self):
    assert CAP_STYLE.round == 1
    assert CAP_STYLE.round == BufferCapStyle.round
    assert CAP_STYLE.flat == 2
    assert CAP_STYLE.flat == BufferCapStyle.flat
    assert CAP_STYLE.square == 3
    assert CAP_STYLE.square == BufferCapStyle.square
    assert JOIN_STYLE.round == 1
    assert JOIN_STYLE.round == BufferJoinStyle.round
    assert JOIN_STYLE.mitre == 2
    assert JOIN_STYLE.mitre == BufferJoinStyle.mitre
    assert JOIN_STYLE.bevel == 3
    assert JOIN_STYLE.bevel == BufferJoinStyle.bevel
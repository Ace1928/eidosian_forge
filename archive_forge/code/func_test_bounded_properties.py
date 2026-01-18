import sys
import pytest
import itertools
from threading import Thread
from kivy.tests.common import GraphicUnitTest, requires_graphics
def test_bounded_properties(self):
    from kivy.graphics.boxshadow import BoxShadow
    bs = BoxShadow()
    bs.pos = (50, 50)
    bs.size = (150, 150)
    bs.offset = (10, -100)
    bs.blur_radius = -80
    bs.spread_radius = (-200, -100)
    bs.border_radius = (0, 0, 100, 0)
    assert bs.size == (0, 0)
    assert bs.blur_radius == 0
    assert bs.border_radius == tuple(map(lambda value: max(1.0, min(value, min(bs.size) / 2)), bs.border_radius))
    bs = BoxShadow(pos=(50, 50), size=(150, 150), offset=(10, -100), blur_radius=-80, spread_radius=(-200, -100), border_radius=(0, 0, 100, 0))
    assert bs.size == (0, 0)
    assert bs.blur_radius == 0
    assert bs.border_radius == tuple(map(lambda value: max(1.0, min(value, min(bs.size) / 2)), bs.border_radius))
import unittest
import pytest
from kivy.event import EventDispatcher
from functools import partial
@pytest.mark.parametrize('set_name', [True, False])
def test_color_property(self, set_name):
    from kivy.properties import ColorProperty
    color = ColorProperty()
    if set_name:
        color.set_name(wid, 'color')
        color.link_eagerly(wid)
    else:
        color.link(wid, 'color')
        color.link_deps(wid, 'color')
    self.assertEqual(color.get(wid), [1, 1, 1, 1])
    color2 = ColorProperty()
    if set_name:
        color2.set_name(wid, 'color2')
        color2.link_eagerly(wid)
    else:
        color2.link(wid, 'color2')
        color2.link_deps(wid, 'color2')
    self.assertEqual(color2.get(wid), [1, 1, 1, 1])
    color.set(wid, 'yellow')
    self.assertEqual(color.get(wid), [1.0, 1.0, 0.0, 1.0])
    color.set(wid, '#00ff00')
    self.assertEqual(color.get(wid), [0, 1, 0, 1])
    color.set(wid, '#7f7fff7f')
    self.assertEqual(color.get(wid)[0], 127 / 255.0)
    self.assertEqual(color.get(wid)[1], 127 / 255.0)
    self.assertEqual(color.get(wid)[2], 1)
    self.assertEqual(color.get(wid)[3], 127 / 255.0)
    color.set(wid, (1, 1, 0))
    self.assertEqual(color.get(wid), [1, 1, 0, 1])
    color.set(wid, (1, 1, 0, 0))
    self.assertEqual(color.get(wid), [1, 1, 0, 0])
    color.set(wid, [1, 1, 1, 1])
    color_value = color.get(wid)
    color_value[0] = 0.5
    self.assertEqual(color.get(wid), [0.5, 1, 1, 1])
    self.assertEqual(color2.get(wid), [1, 1, 1, 1])
    color2.set(wid, color.get(wid))
    self.assertEqual(color2.get(wid), [0.5, 1, 1, 1])
    color.set(wid, [1, 1, 1, 1])
    color_value = color.get(wid)
    color_value[:] = [0, 1, 0, 1]
    self.assertEqual(color.get(wid), [0, 1, 0, 1])
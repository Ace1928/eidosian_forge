import unittest
import pytest
from kivy.event import EventDispatcher
from functools import partial
@pytest.mark.parametrize('set_name', [True, False])
def test_bounded_numeric_property(self, set_name):
    from kivy.properties import BoundedNumericProperty
    bnp = BoundedNumericProperty(0.0, min=0.0, max=3.5)
    if set_name:
        bnp.set_name(wid, 'bnp')
        bnp.link_eagerly(wid)
    else:
        bnp.link(wid, 'bnp')
        bnp.link_deps(wid, 'bnp')
    bnp.set(wid, 1)
    bnp.set(wid, 0.0)
    bnp.set(wid, 3.1)
    bnp.set(wid, 3.5)
    self.assertRaises(ValueError, partial(bnp.set, wid, 3.6))
    self.assertRaises(ValueError, partial(bnp.set, wid, -3))
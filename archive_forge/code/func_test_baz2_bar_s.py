import unittest
from traits.api import HasTraits, Instance, Str
def test_baz2_bar_s(self):
    self.assertEqual(self.baz2.bar.s, 'bar')
    self.assertEqual(self.baz2.bar.s, self.baz.bar.s)
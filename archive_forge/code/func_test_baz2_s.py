import unittest
from traits.api import HasTraits, Instance, Str
def test_baz2_s(self):
    self.assertEqual(self.baz2.s, 'baz')
    self.assertEqual(self.baz2.s, self.baz.s)
import unittest
from traits.api import HasTraits, Instance, Str
def test_baz2_shared(self):
    self.assertIsNot(self.baz2.shared, None)
    self.assertIsNot(self.baz2.shared, self.shared2)
    self.assertIs(self.baz2.shared, self.shared)
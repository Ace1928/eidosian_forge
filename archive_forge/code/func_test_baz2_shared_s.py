import unittest
from traits.api import HasTraits, Instance, Str
def test_baz2_shared_s(self):
    self.assertEqual(self.baz2.shared.s, 'shared')
    self.assertEqual(self.baz2.bar.shared.s, 'shared')
    self.assertEqual(self.baz2.bar.foo.shared.s, 'shared')
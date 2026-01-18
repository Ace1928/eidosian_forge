import unittest
from traits.api import HasTraits, Str, Undefined, ReadOnly, Float
def test_initial_value(self):
    b = Bar()
    self.assertEqual(b.name, Undefined)
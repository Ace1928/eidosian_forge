import unittest
from traits.api import BaseInstance, HasStrictTraits, Instance, TraitError
def test_explicitly_prohibit_none(self):
    obj = HasSlices(disallow_none=slice(2, 5))
    self.assertIsNotNone(obj.disallow_none)
    with self.assertRaises(TraitError):
        obj.disallow_none = None
    self.assertIsNotNone(obj.disallow_none)
    obj = HasSlices(also_disallow_none=slice(2, 5))
    self.assertIsNotNone(obj.also_disallow_none)
    with self.assertRaises(TraitError):
        obj.also_disallow_none = None
    self.assertIsNotNone(obj.also_disallow_none)
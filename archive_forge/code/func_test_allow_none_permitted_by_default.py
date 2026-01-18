import unittest
from traits.api import BaseInstance, HasStrictTraits, Instance, TraitError
def test_allow_none_permitted_by_default(self):
    obj = HasSlices(my_slice=slice(2, 5))
    self.assertIsNotNone(obj.my_slice)
    obj.my_slice = None
    self.assertIsNone(obj.my_slice)
    obj = HasSlices(also_my_slice=slice(2, 5))
    self.assertIsNotNone(obj.also_my_slice)
    obj.my_slice = None
    self.assertIsNone(obj.my_slice)
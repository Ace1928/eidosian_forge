import sys
from breezy import tests
from breezy.tests import features
def test__lookup_after_resize(self):
    obj = self.module.SimpleSet()
    k1 = _Hashable(643)
    k2 = _Hashable(643 + 1024)
    obj.add(k1)
    obj.add(k2)
    self.assertLookup(643, k1, obj, k1)
    self.assertLookup(644, k2, obj, k2)
    obj._py_resize(2047)
    self.assertEqual(2048, obj.mask + 1)
    self.assertLookup(643, k1, obj, k1)
    self.assertLookup(643 + 1024, k2, obj, k2)
    obj._py_resize(1023)
    self.assertEqual(1024, obj.mask + 1)
    self.assertLookup(643, k1, obj, k1)
    self.assertLookup(644, k2, obj, k2)
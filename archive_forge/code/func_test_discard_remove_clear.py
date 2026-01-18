import pickle
import pyomo.common.unittest as unittest
from pyomo.common.collections import OrderedSet
def test_discard_remove_clear(self):
    a = OrderedSet([1, 3, 2, 4])
    a.discard(3)
    self.assertEqual(list(a), [1, 2, 4])
    a.discard(3)
    self.assertEqual(list(a), [1, 2, 4])
    a.remove(2)
    self.assertEqual(list(a), [1, 4])
    with self.assertRaisesRegex(KeyError, '2'):
        a.remove(2)
    a.clear()
    self.assertEqual(list(a), [])
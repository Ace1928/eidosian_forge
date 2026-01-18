import unittest
from traits.api import HasTraits, Int, List
def test_inplace_multiply(self):
    foo = MyClass()
    foo.l *= 2
    self.assertEqual(foo.l, [1, 2, 3, 1, 2, 3])
    self.assertEqual(len(foo.l_events), 1)
    event = foo.l_events[0]
    self.assertEqual(event.added, [1, 2, 3])
    self.assertEqual(event.removed, [])
    self.assertEqual(event.index, 3)
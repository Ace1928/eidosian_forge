import unittest
from traits.api import HasTraits, Int, List
def test_assignment_step_slice(self):
    foo = MyClass()
    foo.l = [1, 2, 3]
    foo.l[::2] = [3, 4]
    self.assertEqual(len(foo.l_events), 1)
    event, = foo.l_events
    self.assertEqual(event.index, slice(0, 3, 2))
    self.assertEqual(event.added, [3, 4])
    self.assertEqual(event.removed, [1, 3])
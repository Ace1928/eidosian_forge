import unittest
from traits.api import HasTraits, Int, List
def test_insert_index_invariants(self):
    for index in range(-10, 10):
        foo = MyClass()
        foo.l.insert(index, 1729)
        self.assertEqual(len(foo.l_events), 1)
        event = foo.l_events[0]
        self.assertEqual(event.added, [1729])
        self.assertEqual(event.removed, [])
        self.assertGreaterEqual(event.index, 0)
        self.assertEqual(foo.l[event.index], 1729)
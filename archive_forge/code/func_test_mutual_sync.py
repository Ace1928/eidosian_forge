import unittest
from traits.api import (
from traits.testing.unittest_tools import UnittestTools
def test_mutual_sync(self):
    """ Test that two traits can be mutually synchronized.
        """
    a = A()
    b = B()
    a.sync_trait('t', b)
    b.t = 10
    self.assertEqual(a.t, b.t)
    a.t = 20
    self.assertEqual(b.t, a.t)
    a.sync_trait('t', b, remove=True)
    with self.assertTraitDoesNotChange(a, 't'):
        b.t = 5
    with self.assertTraitDoesNotChange(b, 't'):
        a.t = 7
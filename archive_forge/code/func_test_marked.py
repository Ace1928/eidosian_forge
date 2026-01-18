from __future__ import absolute_import
from twisted.trial.unittest import TestCase
from .._util import synchronized
def test_marked(self):
    """
        A method wrapped with @synchronized is marked as synchronized.
        """
    self.assertEqual(Lockable.check.synchronized, True)
import os
from operator import getitem
from twisted.python.compat import _PYPY
from twisted.python.fakepwd import ShadowDatabase, UserDatabase
from twisted.trial.unittest import TestCase
def test_getpwnamRejectsBytes(self):
    """
        L{getpwnam} rejects a non-L{str} username with an exception.
        """
    exc_type = TypeError
    if _PYPY:
        exc_type = Exception
    self.assertRaises(exc_type, self.database.getpwnam, b'i-am-bytes')
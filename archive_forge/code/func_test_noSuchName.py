import os
from operator import getitem
from twisted.python.compat import _PYPY
from twisted.python.fakepwd import ShadowDatabase, UserDatabase
from twisted.trial.unittest import TestCase
def test_noSuchName(self):
    """
        I{getspnam} raises L{KeyError} when passed a username which does not
        exist in the user database.
        """
    self.assertRaises(KeyError, self.database.getspnam, 'alice')
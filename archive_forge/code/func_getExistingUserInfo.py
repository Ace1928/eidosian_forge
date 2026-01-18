import os
from operator import getitem
from twisted.python.compat import _PYPY
from twisted.python.fakepwd import ShadowDatabase, UserDatabase
from twisted.trial.unittest import TestCase
def getExistingUserInfo(self):
    """
        Read and return the next record from C{self._users}.
        """
    return next(self._users)
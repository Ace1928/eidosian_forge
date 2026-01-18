import os
from operator import getitem
from twisted.python.compat import _PYPY
from twisted.python.fakepwd import ShadowDatabase, UserDatabase
from twisted.trial.unittest import TestCase
def test_getspnam(self):
    """
        L{getspnam} accepts a username and returns the user record associated
        with it.
        """
    for i in range(2):
        username, password, lastChange, min, max, warn, inact, expire, flag = self.getExistingUserInfo()
        entry = self.database.getspnam(username)
        self.assertEqual(entry.sp_nam, username)
        self.assertEqual(entry.sp_pwd, password)
        self.assertEqual(entry.sp_lstchg, lastChange)
        self.assertEqual(entry.sp_min, min)
        self.assertEqual(entry.sp_max, max)
        self.assertEqual(entry.sp_warn, warn)
        self.assertEqual(entry.sp_inact, inact)
        self.assertEqual(entry.sp_expire, expire)
        self.assertEqual(entry.sp_flag, flag)
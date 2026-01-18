import os
import stat
from typing import Dict, Optional
from twisted.enterprise.adbapi import (
from twisted.internet import defer, interfaces, reactor
from twisted.python.failure import Failure
from twisted.python.reflect import requireModule
from twisted.trial import unittest
def test_closeLogError(self):
    """
        L{ConnectionPool._close} logs exceptions.
        """

    class ConnectionCloseRaise:

        def close(self):
            raise RuntimeError('problem!')
    pool = DummyConnectionPool()
    pool._close(ConnectionCloseRaise())
    errors = self.flushLoggedErrors(RuntimeError)
    self.assertEqual(len(errors), 1)
    self.assertEqual(errors[0].value.args[0], 'problem!')
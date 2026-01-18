import os
import stat
from typing import Dict, Optional
from twisted.enterprise.adbapi import (
from twisted.internet import defer, interfaces, reactor
from twisted.python.failure import Failure
from twisted.python.reflect import requireModule
from twisted.trial import unittest
def makePool(self, **newkw):
    """Create a connection pool with additional keyword arguments."""
    args, kw = self.getPoolArgs()
    kw = kw.copy()
    kw.update(newkw)
    return ConnectionPool(*args, **kw)
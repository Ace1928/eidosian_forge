import os
import stat
from typing import Dict, Optional
from twisted.enterprise.adbapi import (
from twisted.internet import defer, interfaces, reactor
from twisted.python.failure import Failure
from twisted.python.reflect import requireModule
from twisted.trial import unittest
def getPoolArgs(self):
    args = ('kinterbasdb',)
    kw = {'database': self.DB_NAME, 'host': '127.0.0.1', 'user': self.DB_USER, 'password': self.DB_PASS}
    return (args, kw)
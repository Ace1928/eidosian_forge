import os
import stat
from typing import Dict, Optional
from twisted.enterprise.adbapi import (
from twisted.internet import defer, interfaces, reactor
from twisted.python.failure import Failure
from twisted.python.reflect import requireModule
from twisted.trial import unittest
def can_connect(self):
    if requireModule('kinterbasdb') is None:
        return False
    try:
        self.startDB()
        self.stopDB()
        return True
    except BaseException:
        return False
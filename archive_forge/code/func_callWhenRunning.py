import os
import stat
from typing import Dict, Optional
from twisted.enterprise.adbapi import (
from twisted.internet import defer, interfaces, reactor
from twisted.python.failure import Failure
from twisted.python.reflect import requireModule
from twisted.trial import unittest
def callWhenRunning(self, function):
    if self._running:
        function()
    else:
        return self.addSystemEventTrigger('after', 'startup', function)
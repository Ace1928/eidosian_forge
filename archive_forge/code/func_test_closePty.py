import errno
import gc
import gzip
import operator
import os
import signal
import stat
import sys
from unittest import SkipTest, skipIf
from io import BytesIO
from zope.interface.verify import verifyObject
from twisted.internet import defer, error, interfaces, protocol, reactor
from twisted.python import procutils, runtime
from twisted.python.compat import networkString
from twisted.python.filepath import FilePath
from twisted.python.log import msg
from twisted.trial import unittest
def test_closePty(self):
    if self.verbose:
        print('starting processes')
    self.createProcesses(usePTY=1)
    reactor.callLater(1, self.close, 0)
    reactor.callLater(2, self.close, 1)
    return self._onClose()
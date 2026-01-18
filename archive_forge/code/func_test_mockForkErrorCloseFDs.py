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
def test_mockForkErrorCloseFDs(self):
    """
        When C{os.fork} raises an exception, the file descriptors created
        before are closed and don't leak.
        """
    self._mockWithForkError()
    self.assertEqual(set(self.mockos.closed), {-1, -4, -6, -2, -3, -5})
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
def test_mockErrorInPipe(self):
    """
        If C{os.pipe} raises an exception after some pipes where created, the
        created pipes are closed and don't leak.
        """
    pipes = [-1, -2, -3, -4]

    def pipe():
        try:
            return (pipes.pop(0), pipes.pop(0))
        except IndexError:
            raise OSError()
    self.mockos.pipe = pipe
    protocol = TrivialProcessProtocol(None)
    self.assertRaises(OSError, reactor.spawnProcess, protocol, None)
    self.assertEqual(self.mockos.actions, [])
    self.assertEqual(set(self.mockos.closed), {-4, -3, -2, -1})
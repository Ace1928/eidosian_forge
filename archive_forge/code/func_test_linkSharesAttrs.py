import os
import re
import struct
from unittest import skipIf
from hamcrest import assert_that, equal_to
from twisted.internet import defer
from twisted.internet.error import ConnectionLost
from twisted.internet.testing import StringTransport
from twisted.protocols import loopback
from twisted.python import components
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
def test_linkSharesAttrs(self):
    d = self.client.makeLink(b'testLink', b'testfile1')
    self._emptyBuffers()

    def _getFirstAttrs(_):
        d = self.client.getAttrs(b'testLink', 1)
        self._emptyBuffers()
        return d

    def _getSecondAttrs(firstAttrs):
        d = self.client.getAttrs(b'testfile1')
        self._emptyBuffers()
        d.addCallback(self.assertEqual, firstAttrs)
        return d
    d.addCallback(_getFirstAttrs)
    return d.addCallback(_getSecondAttrs)
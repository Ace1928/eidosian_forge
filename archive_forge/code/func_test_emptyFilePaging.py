import gc
import os
import sys
import time
import weakref
from collections import deque
from io import BytesIO as StringIO
from typing import Dict
from zope.interface import Interface, implementer
from twisted.cred import checkers, credentials, portal
from twisted.cred.error import UnauthorizedLogin, UnhandledCredentials
from twisted.internet import address, main, protocol, reactor
from twisted.internet.defer import Deferred, gatherResults, succeed
from twisted.internet.error import ConnectionRefusedError
from twisted.internet.testing import _FakeConnector
from twisted.protocols.policies import WrappingFactory
from twisted.python import failure, log
from twisted.python.compat import iterbytes
from twisted.spread import jelly, pb, publish, util
from twisted.trial import unittest
def test_emptyFilePaging(self):
    """
        Test L{util.FilePager}, sending an empty file.
        """
    filenameEmpty = self.mktemp()
    open(filenameEmpty, 'w').close()
    c, s, pump = connectedServerAndClient(test=self)
    pagerizer = FilePagerizer(filenameEmpty, None)
    s.setNameForLocal('bar', pagerizer)
    x = c.remoteForName('bar')
    l = []
    util.getAllPages(x, 'getPages').addCallback(l.append)
    ttl = 10
    while not l and ttl > 0:
        pump.pump()
        ttl -= 1
    if not ttl:
        self.fail('getAllPages timed out')
    self.assertEqual(b''.join(l[0]), b'', 'Pages received not equal to pages sent!')
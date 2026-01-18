import struct
from itertools import chain
from typing import Dict, List, Tuple
from twisted.conch.test.keydata import (
from twisted.conch.test.loopback import LoopbackRelay
from twisted.cred import portal
from twisted.cred.error import UnauthorizedLogin
from twisted.internet import defer, protocol, reactor
from twisted.internet.error import ProcessTerminated
from twisted.python import failure, log
from twisted.python.reflect import requireModule
from twisted.trial import unittest
from twisted.python import components
def cbExited(ignored):
    if self.channel.status != 0:
        log.msg('shell exit status was not 0: %i' % (self.channel.status,))
    self.assertEqual(b''.join(self.channel.received), b'testing the shell!\x00\r\n')
    self.assertTrue(self.channel.eofCalled)
    self.assertTrue(self.realm.avatar._testSession.eof)
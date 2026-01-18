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
def cbShell(ignored):
    self.channel.write(b'testing the shell!\x00')
    self.channel.conn.sendEOF(self.channel)
    return defer.gatherResults([self.channel.onClose, self.realm.avatar._testSession.onClose])
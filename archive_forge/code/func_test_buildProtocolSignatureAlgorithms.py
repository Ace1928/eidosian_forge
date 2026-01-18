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
def test_buildProtocolSignatureAlgorithms(self):
    """
        buildProtocol() sets supportedPublicKeys to the list of supported
        signature algorithms.
        """
    f = factory.SSHFactory()
    f.getPublicKeys = lambda: {b'ssh-rsa': keys.Key.fromString(publicRSA_openssh), b'ssh-dss': keys.Key.fromString(publicDSA_openssh)}
    f.getPrivateKeys = lambda: {b'ssh-rsa': keys.Key.fromString(privateRSA_openssh), b'ssh-dss': keys.Key.fromString(privateDSA_openssh)}
    f.startFactory()
    p = f.buildProtocol(None)
    self.assertEqual([b'rsa-sha2-512', b'rsa-sha2-256', b'ssh-rsa', b'ssh-dss'], p.supportedPublicKeys)
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
def test_buildProtocolKexECDSA(self):
    """
        ECDSA key exchanges are listed with 256 having a higher priority among ECDSA.
        """
    f2 = self.makeSSHFactory()
    p2 = f2.buildProtocol(None)
    self.assertIn(b'ecdh-sha2-nistp256,ecdh-sha2-nistp384,ecdh-sha2-nistp521', b','.join(p2.supportedKeyExchanges))
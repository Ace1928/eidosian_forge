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
def test_buildProtocolWithPrimes(self):
    """
        Group key exchanges are supported when we have the primes database.
        """
    f2 = self.makeSSHFactory(primes={1: (2, 3)})
    p2 = f2.buildProtocol(None)
    self.assertIn(b'diffie-hellman-group-exchange-sha1', p2.supportedKeyExchanges)
    self.assertIn(b'diffie-hellman-group-exchange-sha256', p2.supportedKeyExchanges)
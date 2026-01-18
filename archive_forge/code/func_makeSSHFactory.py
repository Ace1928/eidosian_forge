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
def makeSSHFactory(self, primes=None):
    sshFactory = factory.SSHFactory()
    sshFactory.getPrimes = lambda: primes
    sshFactory.getPublicKeys = lambda: {b'ssh-rsa': keys.Key.fromString(publicRSA_openssh)}
    sshFactory.getPrivateKeys = lambda: {b'ssh-rsa': keys.Key.fromString(privateRSA_openssh)}
    sshFactory.startFactory()
    return sshFactory
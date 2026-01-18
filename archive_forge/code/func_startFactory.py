import random
from itertools import chain
from typing import Dict, List, Optional, Tuple
from twisted.conch import error
from twisted.conch.ssh import _kex, connection, transport, userauth
from twisted.internet import protocol
from twisted.logger import Logger
def startFactory(self) -> None:
    """
        Check for public and private keys.
        """
    if not hasattr(self, 'publicKeys'):
        self.publicKeys = self.getPublicKeys()
    if not hasattr(self, 'privateKeys'):
        self.privateKeys = self.getPrivateKeys()
    if not self.publicKeys or not self.privateKeys:
        raise error.ConchError('no host keys, failing')
    if not hasattr(self, 'primes'):
        self.primes = self.getPrimes()
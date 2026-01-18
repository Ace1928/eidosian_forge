import random
from itertools import chain
from typing import Dict, List, Optional, Tuple
from twisted.conch import error
from twisted.conch.ssh import _kex, connection, transport, userauth
from twisted.internet import protocol
from twisted.logger import Logger
def getDHPrime(self, bits: int) -> Tuple[int, int]:
    """
        Return a tuple of (g, p) for a Diffe-Hellman process, with p being as
        close to C{bits} bits as possible.
        """

    def keyfunc(i: int) -> int:
        return abs(i - bits)
    assert self.primes is not None, 'Factory should have been started by now.'
    primesKeys = sorted(self.primes.keys(), key=keyfunc)
    realBits = primesKeys[0]
    return random.choice(self.primes[realBits])
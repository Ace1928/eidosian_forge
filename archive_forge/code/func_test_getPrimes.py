import os
from unittest import skipIf
from twisted.conch.ssh._kex import getDHGeneratorAndPrime
from twisted.conch.test import keydata
from twisted.python.filepath import FilePath
from twisted.python.reflect import requireModule
from twisted.test.test_process import MockOS
from twisted.trial.unittest import TestCase
def test_getPrimes(self) -> None:
    """
        L{OpenSSHFactory.getPrimes} should return the available primes
        in the moduli directory.
        """
    primes = self.factory.getPrimes()
    self.assertEqual(primes, {2048: [getDHGeneratorAndPrime(b'diffie-hellman-group14-sha1')]})
import logging
import warnings
from rsa._compat import range
import rsa.prime
import rsa.pem
import rsa.common
import rsa.randnum
import rsa.core
def blind(self, message, r):
    """Performs blinding on the message using random number 'r'.

        :param message: the message, as integer, to blind.
        :type message: int
        :param r: the random number to blind with.
        :type r: int
        :return: the blinded message.
        :rtype: int

        The blinding is such that message = unblind(decrypt(blind(encrypt(message))).

        See https://en.wikipedia.org/wiki/Blinding_%28cryptography%29
        """
    return message * pow(r, self.e, self.n) % self.n
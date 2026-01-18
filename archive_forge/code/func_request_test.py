import struct
from twisted.conch.ssh import channel
from twisted.conch.test import test_userauth
from twisted.python.reflect import requireModule
from twisted.trial import unittest
from twisted.conch import error
def request_test(self, data):
    """
        A test request.  Return True if data is 'data'.

        @type data: L{bytes}
        """
    self.numberRequests += 1
    return data == b'data'
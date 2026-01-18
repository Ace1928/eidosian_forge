import struct
from twisted.conch.ssh import channel
from twisted.conch.test import test_userauth
from twisted.python.reflect import requireModule
from twisted.trial import unittest
from twisted.conch import error
def global_Test_Data(self, data):
    """
        The other side made the 'Test-Data' global request.  Return True and
        the data we received.
        """
    return (True, data)
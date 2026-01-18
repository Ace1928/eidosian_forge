import struct
from twisted.conch.ssh import channel
from twisted.conch.test import test_userauth
from twisted.python.reflect import requireModule
from twisted.trial import unittest
from twisted.conch import error
def test_linkAvatar(self):
    """
        Test that the connection links itself to the avatar in the
        transport.
        """
    self.assertIs(self.transport.avatar.conn, self.conn)
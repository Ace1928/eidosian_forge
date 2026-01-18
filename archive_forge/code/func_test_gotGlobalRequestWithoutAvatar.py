import struct
from twisted.conch.ssh import channel
from twisted.conch.test import test_userauth
from twisted.python.reflect import requireModule
from twisted.trial import unittest
from twisted.conch import error
def test_gotGlobalRequestWithoutAvatar(self):
    """
        Test that gotGlobalRequests dispatches to global_* without an avatar.
        """
    del self.transport.avatar
    self.assertTrue(self.conn.gotGlobalRequest(b'TestGlobal', b'data'))
    self.assertEqual(self.conn.gotGlobalRequest(b'Test-Data', b'data'), (True, b'data'))
    self.assertFalse(self.conn.gotGlobalRequest(b'BadGlobal', b'data'))
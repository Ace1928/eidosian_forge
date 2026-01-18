from __future__ import annotations
from unittest import skipIf
from zope.interface.verify import verifyObject
from twisted.trial.unittest import TestCase
def test_closeReceieved(self) -> None:
    """
        Test that the default closeReceieved closes the connection.
        """
    self.assertFalse(self.channel.closing)
    self.channel.closeReceived()
    self.assertTrue(self.channel.closing)
from twisted.trial import unittest
from twisted.words.protocols.jabber import jid
def test_doubleAt(self) -> None:
    """
        Test for failure on double @ signs.

        This should fail because @ is not a valid character for the host
        part of the JID.
        """
    self.assertRaises(jid.InvalidFormat, jid.parse, 'user@@host')
from twisted.trial import unittest
from twisted.words.protocols.jabber import jid
def test_noHost(self) -> None:
    """
        Test for failure on no host part.
        """
    self.assertRaises(jid.InvalidFormat, jid.parse, 'user@')
from twisted.trial import unittest
from twisted.words.protocols.jabber import jid
def test_inequality(self) -> None:
    """
        Test JID inequality.
        """
    j1 = jid.JID('user1@host/resource')
    j2 = jid.JID('user2@host/resource')
    self.assertNotEqual(j1, j2)
from twisted.trial import unittest
from twisted.words.protocols.jabber import jid
def test_hashable(self) -> None:
    """
        Test JID hashability.
        """
    j1 = jid.JID('user@host/resource')
    j2 = jid.JID('user@host/resource')
    self.assertEqual(hash(j1), hash(j2))
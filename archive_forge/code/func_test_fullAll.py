from twisted.trial import unittest
from twisted.words.protocols.jabber import jid
def test_fullAll(self) -> None:
    """
        Test giving a string representation of the JID.
        """
    j = jid.JID(tuple=('user', 'host', 'resource'))
    self.assertEqual('user@host/resource', j.full())
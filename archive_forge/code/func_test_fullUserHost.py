from twisted.trial import unittest
from twisted.words.protocols.jabber import jid
def test_fullUserHost(self) -> None:
    """
        Test giving a string representation of the JID with user, host.
        """
    j = jid.JID(tuple=('user', 'host', None))
    self.assertEqual('user@host', j.full())
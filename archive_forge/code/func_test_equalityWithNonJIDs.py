from twisted.trial import unittest
from twisted.words.protocols.jabber import jid
def test_equalityWithNonJIDs(self) -> None:
    """
        Test JID equality.
        """
    j = jid.JID('user@host/resource')
    self.assertFalse(j == 'user@host/resource')
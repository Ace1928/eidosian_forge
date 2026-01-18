from twisted.trial import unittest
from twisted.words.protocols.jabber import jid
def test_inequalityWithNonJIDs(self) -> None:
    """
        Test JID equality.
        """
    j = jid.JID('user@host/resource')
    self.assertNotEqual(j, 'user@host/resource')
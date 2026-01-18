from twisted.trial import unittest
from twisted.words.im import basechat, basesupport
def test_contactChangedNickNoKey(self) -> None:
    """
        L{basechat.ChatUI.contactChangedNick} on an
        L{twisted.words.im.interfaces.IPerson} who doesn't have an account
        associated with the L{basechat.ChatUI} instance has no effect.
        """
    self.assertEqual(self.person.name, 'foo')
    self.assertEqual(self.person.account, self.account)
    self.ui.contactChangedNick(self.person, 'bar')
    self.assertEqual(self.person.name, 'foo')
    self.assertEqual(self.person.account, self.account)
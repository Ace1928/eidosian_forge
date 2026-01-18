from twisted.trial import unittest
from twisted.words.im import basechat, basesupport
def test_contactChangedNickNoConversation(self) -> None:
    """
        L{basechat.ChatUI.contactChangedNick} changes the name for an
        L{twisted.words.im.interfaces.IPerson}.
        """
    self.ui.persons[self.person.name, self.person.account] = self.person
    self.assertEqual(self.person.name, 'foo')
    self.assertEqual(self.person.account, self.account)
    self.ui.contactChangedNick(self.person, 'bar')
    self.assertEqual(self.person.name, 'bar')
    self.assertEqual(self.person.account, self.account)
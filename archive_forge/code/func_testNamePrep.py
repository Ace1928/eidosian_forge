from twisted.trial import unittest
from twisted.words.protocols.jabber.xmpp_stringprep import (
def testNamePrep(self) -> None:
    self.assertEqual(nameprep.prepare('example.com'), 'example.com')
    self.assertEqual(nameprep.prepare('Example.com'), 'example.com')
    self.assertRaises(UnicodeError, nameprep.prepare, 'ex@mple.com')
    self.assertRaises(UnicodeError, nameprep.prepare, '-example.com')
    self.assertRaises(UnicodeError, nameprep.prepare, 'example-.com')
    self.assertEqual(nameprep.prepare('straße.example.com'), 'strasse.example.com')
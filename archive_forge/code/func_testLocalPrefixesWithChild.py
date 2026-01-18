from zope.interface.verify import verifyObject
from twisted.python.reflect import requireModule
from twisted.trial import unittest
from twisted.words.xish import domish
def testLocalPrefixesWithChild(self):
    e = domish.Element(('testns', 'foo'), localPrefixes={'bar': 'testns'})
    e.addElement('baz')
    self.assertIdentical(e.baz.defaultUri, None)
    self.assertEqual(e.toXml(), "<bar:foo xmlns:bar='testns'><baz/></bar:foo>")
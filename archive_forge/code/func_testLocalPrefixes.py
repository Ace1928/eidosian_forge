from zope.interface.verify import verifyObject
from twisted.python.reflect import requireModule
from twisted.trial import unittest
from twisted.words.xish import domish
def testLocalPrefixes(self):
    e = domish.Element(('testns', 'foo'), localPrefixes={'bar': 'testns'})
    self.assertEqual(e.toXml(), "<bar:foo xmlns:bar='testns'/>")
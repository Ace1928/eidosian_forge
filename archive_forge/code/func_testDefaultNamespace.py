from zope.interface.verify import verifyObject
from twisted.python.reflect import requireModule
from twisted.trial import unittest
from twisted.words.xish import domish
def testDefaultNamespace(self):
    e = domish.Element(('testns', 'foo'))
    self.assertEqual(e.toXml(), "<foo xmlns='testns'/>")
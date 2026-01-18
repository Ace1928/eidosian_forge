from zope.interface.verify import verifyObject
from twisted.python.reflect import requireModule
from twisted.trial import unittest
from twisted.words.xish import domish
def testChildSameNamespace(self):
    e = domish.Element(('testns', 'foo'))
    e.addElement(('testns', 'bar'))
    self.assertEqual(e.toXml(), "<foo xmlns='testns'><bar/></foo>")
from zope.interface.verify import verifyObject
from twisted.python.reflect import requireModule
from twisted.trial import unittest
from twisted.words.xish import domish
def testChildSameDefaultNamespace(self):
    e = domish.Element(('testns', 'foo'))
    e.addElement('bar', 'testns')
    self.assertEqual(e.toXml(), "<foo xmlns='testns'><bar/></foo>")
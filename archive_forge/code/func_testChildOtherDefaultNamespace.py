from zope.interface.verify import verifyObject
from twisted.python.reflect import requireModule
from twisted.trial import unittest
from twisted.words.xish import domish
def testChildOtherDefaultNamespace(self):
    e = domish.Element(('testns', 'foo'))
    e.addElement(('testns2', 'bar'), 'testns2')
    self.assertEqual(e.toXml(), "<foo xmlns='testns'><bar xmlns='testns2'/></foo>")
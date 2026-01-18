from zope.interface.verify import verifyObject
from twisted.python.reflect import requireModule
from twisted.trial import unittest
from twisted.words.xish import domish
def testChildDefaultNS(self):
    xml = b"<root xmlns='testns'><child/></root>"
    self.stream.parse(xml)
    self.assertEqual(self.root.uri, 'testns')
    self.assertEqual(self.elements[0].uri, 'testns')
from zope.interface.verify import verifyObject
from twisted.python.reflect import requireModule
from twisted.trial import unittest
from twisted.words.xish import domish
def testEmptyChildNS(self):
    xml = b"<root xmlns='testns'>\n                    <child1><child2 xmlns=''/></child1>\n                  </root>"
    self.stream.parse(xml)
    self.assertEqual(self.elements[0].child2.uri, '')
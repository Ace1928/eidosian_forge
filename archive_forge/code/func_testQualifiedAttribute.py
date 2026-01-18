from zope.interface.verify import verifyObject
from twisted.python.reflect import requireModule
from twisted.trial import unittest
from twisted.words.xish import domish
def testQualifiedAttribute(self):
    e = domish.Element((None, 'foo'), attribs={('testns2', 'bar'): 'baz'})
    self.assertEqual(e.toXml(), "<foo xmlns:xn0='testns2' xn0:bar='baz'/>")
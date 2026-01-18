from zope.interface.verify import verifyObject
from twisted.python.reflect import requireModule
from twisted.trial import unittest
from twisted.words.xish import domish
def testOtherNamespace(self):
    e = domish.Element(('testns', 'foo'), 'testns2')
    self.assertEqual(e.toXml({'testns': 'bar'}), "<bar:foo xmlns:bar='testns' xmlns='testns2'/>")
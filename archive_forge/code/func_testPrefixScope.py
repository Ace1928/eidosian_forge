from zope.interface.verify import verifyObject
from twisted.python.reflect import requireModule
from twisted.trial import unittest
from twisted.words.xish import domish
def testPrefixScope(self):
    e = domish.Element(('testns', 'foo'))
    self.assertEqual(e.toXml(prefixes={'testns': 'bar'}, prefixesInScope=['bar']), '<bar:foo/>')
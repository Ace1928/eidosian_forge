from zope.interface.verify import verifyObject
from twisted.python.reflect import requireModule
from twisted.trial import unittest
from twisted.words.xish import domish
def testDefaultNSPrefix(self):
    e = domish.Element((None, 'foo'), attribs={('testns2', 'bar'): 'baz'})
    c = e.addElement(('testns2', 'qux'))
    c['testns2', 'bar'] = 'quux'
    c.addElement('foo')
    self.assertEqual(e.toXml(), "<foo xmlns:xn0='testns2' xn0:bar='baz'><xn0:qux xn0:bar='quux'><xn0:foo/></xn0:qux></foo>")
from zope.interface.verify import verifyObject
from twisted.python.reflect import requireModule
from twisted.trial import unittest
from twisted.words.xish import domish
def testTwoChilds(self):
    e = domish.Element(('', 'foo'))
    child1 = e.addElement(('testns', 'bar'), 'testns2')
    child1.addElement(('testns2', 'quux'))
    child2 = e.addElement(('testns3', 'baz'), 'testns4')
    child2.addElement(('testns', 'quux'))
    self.assertEqual(e.toXml(), "<foo><xn0:bar xmlns:xn0='testns' xmlns='testns2'><quux/></xn0:bar><xn1:baz xmlns:xn1='testns3' xmlns='testns4'><xn0:quux xmlns:xn0='testns'/></xn1:baz></foo>")
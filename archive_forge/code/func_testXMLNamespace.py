from zope.interface.verify import verifyObject
from twisted.python.reflect import requireModule
from twisted.trial import unittest
from twisted.words.xish import domish
def testXMLNamespace(self):
    e = domish.Element((None, 'foo'), attribs={('http://www.w3.org/XML/1998/namespace', 'lang'): 'en_US'})
    self.assertEqual(e.toXml(), "<foo xml:lang='en_US'/>")
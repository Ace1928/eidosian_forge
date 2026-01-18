from zope.interface.verify import verifyObject
from twisted.python.reflect import requireModule
from twisted.trial import unittest
from twisted.words.xish import domish
def testRawXMLWithUnicodeSerialization(self):
    e = domish.Element((None, 'foo'))
    e.addRawXml('<degree>°</degree>')
    self.assertEqual(e.toXml(), '<foo><degree>°</degree></foo>')
from zope.interface.verify import verifyObject
from twisted.python.reflect import requireModule
from twisted.trial import unittest
from twisted.words.xish import domish
def test_addContentNativeStringASCII(self):
    """
        ASCII native strings passed to C{addContent} become the character data.
        """
    element = domish.Element(('testns', 'foo'))
    element.addContent('native')
    self.assertEqual('native', str(element))
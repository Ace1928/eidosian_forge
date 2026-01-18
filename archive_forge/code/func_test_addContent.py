from zope.interface.verify import verifyObject
from twisted.python.reflect import requireModule
from twisted.trial import unittest
from twisted.words.xish import domish
def test_addContent(self):
    """
        Unicode strings passed to C{addContent} become the character data.
        """
    element = domish.Element(('testns', 'foo'))
    element.addContent('unicode')
    self.assertEqual('unicode', str(element))
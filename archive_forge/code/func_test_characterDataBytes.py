from zope.interface.verify import verifyObject
from twisted.python.reflect import requireModule
from twisted.trial import unittest
from twisted.words.xish import domish
def test_characterDataBytes(self):
    """
        Extract character data as UTF-8 using L{bytes}.
        """
    element = domish.Element(('testns', 'foo'))
    element.addContent('☃')
    text = bytes(element)
    self.assertEqual('☃'.encode(), text)
    self.assertIsInstance(text, bytes)
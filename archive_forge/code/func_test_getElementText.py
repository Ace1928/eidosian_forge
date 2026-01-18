from twisted.trial import unittest
from twisted.words.protocols.jabber import error
from twisted.words.xish import domish
def test_getElementText(self) -> None:
    """
        Test getting an element for an error with a text.
        """
    e = error.BaseError('feature-not-implemented', 'text')
    element = e.getElement()
    self.assertEqual(len(element.children), 2)
    self.assertEqual(str(element.text), 'text')
    self.assertEqual(element.text.getAttribute((NS_XML, 'lang')), None)
from twisted.trial import unittest
from twisted.words.protocols.jabber import error
from twisted.words.xish import domish
def test_getElementType(self) -> None:
    """
        Test getting an element for a stanza error with a given type.
        """
    e = error.StanzaError('feature-not-implemented', 'auth')
    element = e.getElement()
    self.assertEqual(element.uri, None)
    self.assertEqual(element['type'], 'auth')
    self.assertEqual(element['code'], '501')
from twisted.trial import unittest
from twisted.words.protocols.jabber import error
from twisted.words.xish import domish
def test_getElementPlain(self) -> None:
    """
        Test getting an element for a plain stanza error.
        """
    e = error.StanzaError('feature-not-implemented')
    element = e.getElement()
    self.assertEqual(element.uri, None)
    self.assertEqual(element['type'], 'cancel')
    self.assertEqual(element['code'], '501')
from twisted.trial import unittest
from twisted.words.protocols.jabber import error
from twisted.words.xish import domish
def test_getElementConditionNamespace(self) -> None:
    """
        Test that the error condition element has the correct namespace.
        """
    e = error.StanzaError('feature-not-implemented')
    element = e.getElement()
    self.assertEqual(NS_XMPP_STANZAS, getattr(element, 'feature-not-implemented').uri)
from hashlib import sha1
from zope.interface.verify import verifyObject
from twisted.python import failure
from twisted.trial import unittest
from twisted.words.protocols.jabber import component, ijabber, xmlstream
from twisted.words.protocols.jabber.jid import JID
from twisted.words.xish import domish
from twisted.words.xish.utility import XmlPipe
def test_addRoute(self):
    """
        Test route registration and routing on incoming stanzas.
        """
    router = component.Router()
    routed = []
    router.route = lambda element: routed.append(element)
    pipe = XmlPipe()
    router.addRoute('example.org', pipe.sink)
    self.assertEqual(1, len(router.routes))
    self.assertEqual(pipe.sink, router.routes['example.org'])
    element = domish.Element(('testns', 'test'))
    pipe.source.send(element)
    self.assertEqual([element], routed)
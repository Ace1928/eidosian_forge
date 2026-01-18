from zope.interface import implementer
from twisted.application import service
from twisted.internet import defer
from twisted.python import log
from twisted.words.protocols.jabber import error, ijabber, jstrports, xmlstream
from twisted.words.protocols.jabber.jid import internJID as JID
from twisted.words.xish import domish
def onElement(self, element):
    """
        Called on incoming XML Stanzas.

        The very first element received should be a request for handshake.
        Otherwise, the stream is dropped with a 'not-authorized' error. If a
        handshake request was received, the hash is extracted and passed to
        L{onHandshake}.
        """
    if (element.uri, element.name) == (self.namespace, 'handshake'):
        self.onHandshake(str(element))
    else:
        exc = error.StreamError('not-authorized')
        self.xmlstream.sendStreamError(exc)
from zope.interface import implementer
from twisted.application import service
from twisted.internet import defer
from twisted.python import log
from twisted.words.protocols.jabber import error, ijabber, jstrports, xmlstream
from twisted.words.protocols.jabber.jid import internJID as JID
from twisted.words.xish import domish
def streamStarted(self, rootElement):
    """
        Called by the stream when it has started.

        This examines the default namespace of the incoming stream and whether
        there is a requested hostname for the component. Then it generates a
        stream identifier, sends a response header and adds an observer for
        the first incoming element, triggering L{onElement}.
        """
    xmlstream.ListenAuthenticator.streamStarted(self, rootElement)
    if rootElement.defaultUri != self.namespace:
        exc = error.StreamError('invalid-namespace')
        self.xmlstream.sendStreamError(exc)
        return
    if not self.xmlstream.thisEntity:
        exc = error.StreamError('improper-addressing')
        self.xmlstream.sendStreamError(exc)
        return
    self.xmlstream.sendHeader()
    self.xmlstream.addOnetimeObserver('/*', self.onElement)
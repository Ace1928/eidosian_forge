from zope.interface import implementer
from twisted.application import service
from twisted.internet import defer
from twisted.python import log
from twisted.words.protocols.jabber import error, ijabber, jstrports, xmlstream
from twisted.words.protocols.jabber.jid import internJID as JID
from twisted.words.xish import domish
def componentFactory(componentid, password):
    """
    XML stream factory for external server-side components.

    @param componentid: JID of the component.
    @type componentid: L{unicode}
    @param password: password used to authenticate to the server.
    @type password: C{str}
    """
    a = ConnectComponentAuthenticator(componentid, password)
    return xmlstream.XmlStreamFactory(a)
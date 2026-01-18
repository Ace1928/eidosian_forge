from zope.interface import implementer
from twisted.application import service
from twisted.internet import defer
from twisted.python import log
from twisted.words.protocols.jabber import error, ijabber, jstrports, xmlstream
from twisted.words.protocols.jabber.jid import internJID as JID
from twisted.words.xish import domish
def buildServiceManager(jid, password, strport):
    """
    Constructs a pre-built L{ServiceManager}, using the specified strport
    string.
    """
    svc = ServiceManager(jid, password)
    client_svc = jstrports.client(strport, svc.getFactory())
    client_svc.setServiceParent(svc)
    return svc
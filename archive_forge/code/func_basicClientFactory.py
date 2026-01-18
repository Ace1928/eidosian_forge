from twisted.words.protocols.jabber import error, sasl, xmlstream
from twisted.words.protocols.jabber.jid import JID
from twisted.words.xish import domish, utility, xpath
def basicClientFactory(jid, secret):
    a = BasicAuthenticator(jid, secret)
    return xmlstream.XmlStreamFactory(a)
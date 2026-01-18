from twisted.words.protocols.jabber import error, sasl, xmlstream
from twisted.words.protocols.jabber.jid import JID
from twisted.words.xish import domish, utility, xpath
def onBind(self, iq):
    if iq.bind:
        self.xmlstream.authenticator.jid = JID(str(iq.bind.jid))
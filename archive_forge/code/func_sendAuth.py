import re
from base64 import b64decode, b64encode
from twisted.internet import defer
from twisted.words.protocols.jabber import sasl_mechanisms, xmlstream
from twisted.words.xish import domish
def sendAuth(self, data=None):
    """
        Initiate authentication protocol exchange.

        If an initial client response is given in C{data}, it will be
        sent along.

        @param data: initial client response.
        @type data: C{str} or L{None}.
        """
    auth = domish.Element((NS_XMPP_SASL, 'auth'))
    auth['mechanism'] = self.mechanism.name
    if data is not None:
        auth.addContent(b64encode(data).decode('ascii') or '=')
    self.xmlstream.send(auth)
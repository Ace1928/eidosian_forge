import re
from base64 import b64decode, b64encode
from twisted.internet import defer
from twisted.words.protocols.jabber import sasl_mechanisms, xmlstream
from twisted.words.xish import domish
def setMechanism(self):
    """
        Select and setup authentication mechanism.

        Uses the authenticator's C{jid} and C{password} attribute for the
        authentication credentials. If no supported SASL mechanisms are
        advertized by the receiving party, a failing deferred is returned with
        a L{SASLNoAcceptableMechanism} exception.
        """
    jid = self.xmlstream.authenticator.jid
    password = self.xmlstream.authenticator.password
    mechanisms = get_mechanisms(self.xmlstream)
    if jid.user is not None:
        if 'DIGEST-MD5' in mechanisms:
            self.mechanism = sasl_mechanisms.DigestMD5('xmpp', jid.host, None, jid.user, password)
        elif 'PLAIN' in mechanisms:
            self.mechanism = sasl_mechanisms.Plain(None, jid.user, password)
        else:
            raise SASLNoAcceptableMechanism()
    elif 'ANONYMOUS' in mechanisms:
        self.mechanism = sasl_mechanisms.Anonymous()
    else:
        raise SASLNoAcceptableMechanism()
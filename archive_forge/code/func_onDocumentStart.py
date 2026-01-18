from binascii import hexlify
from hashlib import sha1
from sys import intern
from typing import Optional, Tuple
from zope.interface import directlyProvides, implementer
from twisted.internet import defer, protocol
from twisted.internet.error import ConnectionLost
from twisted.python import failure, log, randbytes
from twisted.words.protocols.jabber import error, ijabber, jid
from twisted.words.xish import domish, xmlstream
from twisted.words.xish.xmlstream import (
def onDocumentStart(self, rootElement):
    """
        Called when the stream header has been received.

        Extracts the header's C{id} and C{version} attributes from the root
        element. The C{id} attribute is stored in our C{sid} attribute and the
        C{version} attribute is parsed and the minimum of the version we sent
        and the parsed C{version} attribute is stored as a tuple (major, minor)
        in this class' C{version} attribute. If no C{version} attribute was
        present, we assume version 0.0.

        If appropriate (we are the initiating stream and the minimum of our and
        the other party's version is at least 1.0), a one-time observer is
        registered for getting the stream features. The registered function is
        C{onFeatures}.

        Ultimately, the authenticator's C{streamStarted} method will be called.

        @param rootElement: The root element.
        @type rootElement: L{domish.Element}
        """
    xmlstream.XmlStream.onDocumentStart(self, rootElement)
    self.addOnetimeObserver("/error[@xmlns='%s']" % NS_STREAMS, self.onStreamError)
    self.authenticator.streamStarted(rootElement)
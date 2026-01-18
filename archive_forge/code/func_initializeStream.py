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
def initializeStream(self):
    """
        Perform stream initialization procedures.

        An L{XmlStream} holds a list of initializer objects in its
        C{initializers} attribute. This method calls these initializers in
        order and dispatches the L{STREAM_AUTHD_EVENT} event when the list has
        been successfully processed. Otherwise it dispatches the
        C{INIT_FAILED_EVENT} event with the failure.

        Initializers may return the special L{Reset} object to halt the
        initialization processing. It signals that the current initializer was
        successfully processed, but that the XML Stream has been reset. An
        example is the TLSInitiatingInitializer.
        """

    def remove_first(result):
        self.xmlstream.initializers.pop(0)
        return result

    def do_next(result):
        """
            Take the first initializer and process it.

            On success, the initializer is removed from the list and
            then next initializer will be tried.
            """
        if result is Reset:
            return None
        try:
            init = self.xmlstream.initializers[0]
        except IndexError:
            self.xmlstream.dispatch(self.xmlstream, STREAM_AUTHD_EVENT)
            return None
        else:
            d = defer.maybeDeferred(init.initialize)
            d.addCallback(remove_first)
            d.addCallback(do_next)
            return d
    d = defer.succeed(None)
    d.addCallback(do_next)
    d.addErrback(self.xmlstream.dispatch, INIT_FAILED_EVENT)
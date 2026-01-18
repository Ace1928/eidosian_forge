from typing import TYPE_CHECKING, Callable, List, Optional
from zope.interface import Attribute, Interface
from twisted.cred.credentials import IUsernameDigestHash
from twisted.internet.defer import Deferred
from twisted.internet.interfaces import IPushProducer
from twisted.web.http_headers import Headers
def prePathURL():
    """
        At any time during resource traversal or resource rendering,
        returns an absolute URL to the most nested resource which has
        yet been reached.

        @see: {twisted.web.server.Request.prepath}

        @return: An absolute URL.
        @rtype: L{bytes}
        """
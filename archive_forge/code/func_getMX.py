import email.utils
import os
import pickle
import time
from typing import Type
from twisted.application import internet
from twisted.internet import protocol
from twisted.internet.defer import Deferred, DeferredList
from twisted.internet.error import DNSLookupError
from twisted.internet.protocol import connectionDone
from twisted.mail import bounce, relay, smtp
from twisted.python import log
from twisted.python.failure import Failure
def getMX(self, domain, maximumCanonicalChainLength=3):
    """
        Find the name of a host that acts as a mail exchange server
        for a domain.

        @type domain: L{bytes}
        @param domain: A domain name.

        @type maximumCanonicalChainLength: L{int}
        @param maximumCanonicalChainLength: The maximum number of unique
            canonical name records to follow while looking up the mail exchange
            host.

        @rtype: L{Deferred} which successfully fires with L{Record_MX}
        @return: A deferred which succeeds with the MX record for the mail
            exchange server for the domain or fails if none can be found.
        """
    mailExchangeDeferred = self.resolver.lookupMailExchange(domain)
    mailExchangeDeferred.addCallback(self._filterRecords)
    mailExchangeDeferred.addCallback(self._cbMX, domain, maximumCanonicalChainLength)
    mailExchangeDeferred.addErrback(self._ebMX, domain)
    return mailExchangeDeferred
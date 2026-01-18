import time
from twisted.internet import protocol
from twisted.names import dns, resolve
from twisted.python import log
def gotResolverError(self, failure, protocol, message, address):
    """
        A callback used by L{DNSServerFactory.handleQuery} for handling deferred
        errors from C{self.resolver.query}.

        Constructs a response message from the original query message by
        assigning a suitable error code to C{rCode}.

        An error message will be logged if C{DNSServerFactory.verbose} is C{>1}.

        @param failure: The reason for the failed resolution (as reported by
            C{self.resolver.query}).
        @type failure: L{Failure<twisted.python.failure.Failure>}

        @param protocol: The DNS protocol instance to which to send a response
            message.
        @type protocol: L{dns.DNSDatagramProtocol} or L{dns.DNSProtocol}

        @param message: The original DNS query message for which a response
            message will be constructed.
        @type message: L{dns.Message}

        @param address: The address to which the response message will be sent
            or L{None} if C{protocol} is a stream protocol.
        @type address: L{tuple} or L{None}
        """
    if failure.check(dns.DomainError, dns.AuthoritativeDomainError):
        rCode = dns.ENAME
    else:
        rCode = dns.ESERVER
        log.err(failure)
    response = self._responseFromMessage(message=message, rCode=rCode)
    self.sendReply(protocol, response, address)
    self._verboseLog('Lookup failed')
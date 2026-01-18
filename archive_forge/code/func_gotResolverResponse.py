import time
from twisted.internet import protocol
from twisted.names import dns, resolve
from twisted.python import log
def gotResolverResponse(self, response, protocol, message, address):
    """
        A callback used by L{DNSServerFactory.handleQuery} for handling the
        deferred response from C{self.resolver.query}.

        Constructs a response message by combining the original query message
        with the resolved answer, authority and additional records.

        Marks the response message as authoritative if any of the resolved
        answers are found to be authoritative.

        The resolved answers count will be logged if C{DNSServerFactory.verbose}
        is C{>1}.

        @param response: Answer records, authority records and additional records
        @type response: L{tuple} of L{list} of L{dns.RRHeader} instances

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
    ans, auth, add = response
    response = self._responseFromMessage(message=message, rCode=dns.OK, answers=ans, authority=auth, additional=add)
    self.sendReply(protocol, response, address)
    l = len(ans) + len(auth) + len(add)
    self._verboseLog('Lookup found %d record%s' % (l, l != 1 and 's' or ''))
    if self.cache and l:
        self.cache.cacheResult(message.queries[0], (ans, auth, add))
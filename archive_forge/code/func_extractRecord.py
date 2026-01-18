import socket
from zope.interface import implementer
from twisted.internet import defer, error, interfaces
from twisted.logger import Logger
from twisted.names import dns
from twisted.names.error import (
def extractRecord(resolver, name, answers, level=10):
    """
    Resolve a name to an IP address, following I{CNAME} records and I{NS}
    referrals recursively.

    This is an implementation detail of L{ResolverBase.getHostByName}.

    @param resolver: The resolver to use for the next query (unless handling
    an I{NS} referral).
    @type resolver: L{IResolver}

    @param name: The name being looked up.
    @type name: L{dns.Name}

    @param answers: All of the records returned by the previous query (answers,
    authority, and additional concatenated).
    @type answers: L{list} of L{dns.RRHeader}

    @param level: Remaining recursion budget. This is decremented at each
    recursion. The query returns L{None} when it reaches 0.
    @type level: L{int}

    @returns: The first IPv4 or IPv6 address (as a dotted quad or colon
    quibbles), or L{None} when no result is found.
    @rtype: native L{str} or L{None}
    """
    if not level:
        return None
    if hasattr(socket, 'inet_ntop'):
        for r in answers:
            if r.name == name and r.type == dns.A6:
                return socket.inet_ntop(socket.AF_INET6, r.payload.address)
        for r in answers:
            if r.name == name and r.type == dns.AAAA:
                return socket.inet_ntop(socket.AF_INET6, r.payload.address)
    for r in answers:
        if r.name == name and r.type == dns.A:
            return socket.inet_ntop(socket.AF_INET, r.payload.address)
    for r in answers:
        if r.name == name and r.type == dns.CNAME:
            result = extractRecord(resolver, r.payload.name, answers, level - 1)
            if not result:
                return resolver.getHostByName(r.payload.name.name, effort=level - 1)
            return result
    for r in answers:
        if r.type != dns.NS:
            continue
        from twisted.names import client
        nsResolver = client.Resolver(servers=[(r.payload.name.name.decode('ascii'), dns.PORT)])

        def queryAgain(records):
            ans, auth, add = records
            return extractRecord(nsResolver, name, ans + auth + add, level - 1)
        return nsResolver.lookupAddress(name.name).addCallback(queryAgain)
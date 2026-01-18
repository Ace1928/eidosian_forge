from __future__ import generators
import sys
import re
import os
from io import BytesIO
import dns.exception
import dns.name
import dns.node
import dns.rdataclass
import dns.rdatatype
import dns.rdata
import dns.rdtypes.ANY.SOA
import dns.rrset
import dns.tokenizer
import dns.ttl
import dns.grange
from ._compat import string_types, text_type, PY3
def from_xfr(xfr, zone_factory=Zone, relativize=True, check_origin=True):
    """Convert the output of a zone transfer generator into a zone object.

    @param xfr: The xfr generator
    @type xfr: generator of dns.message.Message objects
    @param relativize: should names be relativized?  The default is True.
    It is essential that the relativize setting matches the one specified
    to dns.query.xfr().
    @type relativize: bool
    @param check_origin: should sanity checks of the origin node be done?
    The default is True.
    @type check_origin: bool
    @raises dns.zone.NoSOA: No SOA RR was found at the zone origin
    @raises dns.zone.NoNS: No NS RRset was found at the zone origin
    @rtype: dns.zone.Zone object
    """
    z = None
    for r in xfr:
        if z is None:
            if relativize:
                origin = r.origin
            else:
                origin = r.answer[0].name
            rdclass = r.answer[0].rdclass
            z = zone_factory(origin, rdclass, relativize=relativize)
        for rrset in r.answer:
            znode = z.nodes.get(rrset.name)
            if not znode:
                znode = z.node_factory()
                z.nodes[rrset.name] = znode
            zrds = znode.find_rdataset(rrset.rdclass, rrset.rdtype, rrset.covers, True)
            zrds.update_ttl(rrset.ttl)
            for rd in rrset:
                rd.choose_relativity(z.origin, relativize)
                zrds.add(rd)
    if check_origin:
        z.check_origin()
    return z
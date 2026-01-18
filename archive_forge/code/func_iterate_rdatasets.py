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
def iterate_rdatasets(self, rdtype=dns.rdatatype.ANY, covers=dns.rdatatype.NONE):
    """Return a generator which yields (name, rdataset) tuples for
        all rdatasets in the zone which have the specified I{rdtype}
        and I{covers}.  If I{rdtype} is dns.rdatatype.ANY, the default,
        then all rdatasets will be matched.

        @param rdtype: int or string
        @type rdtype: int or string
        @param covers: the covered type (defaults to None)
        @type covers: int or string
        """
    if isinstance(rdtype, string_types):
        rdtype = dns.rdatatype.from_text(rdtype)
    if isinstance(covers, string_types):
        covers = dns.rdatatype.from_text(covers)
    for name, node in self.iteritems():
        for rds in node:
            if rdtype == dns.rdatatype.ANY or (rds.rdtype == rdtype and rds.covers == covers):
                yield (name, rds)
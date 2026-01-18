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
def check_origin(self):
    """Do some simple checking of the zone's origin.

        @raises dns.zone.NoSOA: there is no SOA RR
        @raises dns.zone.NoNS: there is no NS RRset
        @raises KeyError: there is no origin node
        """
    if self.relativize:
        name = dns.name.empty
    else:
        name = self.origin
    if self.get_rdataset(name, dns.rdatatype.SOA) is None:
        raise NoSOA
    if self.get_rdataset(name, dns.rdatatype.NS) is None:
        raise NoNS
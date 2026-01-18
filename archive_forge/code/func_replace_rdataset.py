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
def replace_rdataset(self, name, replacement):
    """Replace an rdataset at name.

        It is not an error if there is no rdataset matching I{replacement}.

        Ownership of the I{replacement} object is transferred to the zone;
        in other words, this method does not store a copy of I{replacement}
        at the node, it stores I{replacement} itself.

        If the I{name} node does not exist, it is created.

        @param name: the owner name
        @type name: DNS.name.Name object or string
        @param replacement: the replacement rdataset
        @type replacement: dns.rdataset.Rdataset
        """
    if replacement.rdclass != self.rdclass:
        raise ValueError('replacement.rdclass != zone.rdclass')
    node = self.find_node(name, True)
    node.replace_rdataset(replacement)
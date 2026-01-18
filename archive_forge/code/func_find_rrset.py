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
def find_rrset(self, name, rdtype, covers=dns.rdatatype.NONE):
    """Look for rdata with the specified name and type in the zone,
        and return an RRset encapsulating it.

        The I{name}, I{rdtype}, and I{covers} parameters may be
        strings, in which case they will be converted to their proper
        type.

        This method is less efficient than the similar
        L{find_rdataset} because it creates an RRset instead of
        returning the matching rdataset.  It may be more convenient
        for some uses since it returns an object which binds the owner
        name to the rdata.

        This method may not be used to create new nodes or rdatasets;
        use L{find_rdataset} instead.

        KeyError is raised if the name or type are not found.
        Use L{get_rrset} if you want to have None returned instead.

        @param name: the owner name to look for
        @type name: DNS.name.Name object or string
        @param rdtype: the rdata type desired
        @type rdtype: int or string
        @param covers: the covered type (defaults to None)
        @type covers: int or string
        @raises KeyError: the node or rdata could not be found
        @rtype: dns.rrset.RRset object
        """
    name = self._validate_name(name)
    if isinstance(rdtype, string_types):
        rdtype = dns.rdatatype.from_text(rdtype)
    if isinstance(covers, string_types):
        covers = dns.rdatatype.from_text(covers)
    rdataset = self.nodes[name].find_rdataset(self.rdclass, rdtype, covers)
    rrset = dns.rrset.RRset(name, self.rdclass, rdtype, covers)
    rrset.update(rdataset)
    return rrset
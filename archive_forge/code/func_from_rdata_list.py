import dns.name
import dns.rdataset
import dns.rdataclass
import dns.renderer
from ._compat import string_types
def from_rdata_list(name, ttl, rdatas, idna_codec=None):
    """Create an RRset with the specified name and TTL, and with
    the specified list of rdata objects.

    Returns a ``dns.rrset.RRset`` object.
    """
    if isinstance(name, string_types):
        name = dns.name.from_text(name, None, idna_codec=idna_codec)
    if len(rdatas) == 0:
        raise ValueError('rdata list must not be empty')
    r = None
    for rd in rdatas:
        if r is None:
            r = RRset(name, rd.rdclass, rd.rdtype)
            r.update_ttl(ttl)
        r.add(rd)
    return r
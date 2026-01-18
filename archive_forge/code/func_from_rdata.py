import dns.name
import dns.rdataset
import dns.rdataclass
import dns.renderer
from ._compat import string_types
def from_rdata(name, ttl, *rdatas):
    """Create an RRset with the specified name and TTL, and with
    the specified rdata objects.

    Returns a ``dns.rrset.RRset`` object.
    """
    return from_rdata_list(name, ttl, rdatas)
from io import BytesIO
import struct
import time
import dns.exception
import dns.name
import dns.node
import dns.rdataset
import dns.rdata
import dns.rdatatype
import dns.rdataclass
from ._compat import string_types
def make_ds(name, key, algorithm, origin=None):
    """Create a DS record for a DNSSEC key.

    *name* is the owner name of the DS record.

    *key* is a ``dns.rdtypes.ANY.DNSKEY``.

    *algorithm* is a string describing which hash algorithm to use.  The
    currently supported hashes are "SHA1" and "SHA256".  Case does not
    matter for these strings.

    *origin* is a ``dns.name.Name`` and will be used as the origin
    if *key* is a relative name.

    Returns a ``dns.rdtypes.ANY.DS``.
    """
    if algorithm.upper() == 'SHA1':
        dsalg = 1
        hash = SHA1.new()
    elif algorithm.upper() == 'SHA256':
        dsalg = 2
        hash = SHA256.new()
    else:
        raise UnsupportedAlgorithm('unsupported algorithm "%s"' % algorithm)
    if isinstance(name, string_types):
        name = dns.name.from_text(name, origin)
    hash.update(name.canonicalize().to_wire())
    hash.update(_to_rdata(key, origin))
    digest = hash.digest()
    dsrdata = struct.pack('!HBB', key_id(key), key.algorithm, dsalg) + digest
    return dns.rdata.from_wire(dns.rdataclass.IN, dns.rdatatype.DS, dsrdata, 0, len(dsrdata))
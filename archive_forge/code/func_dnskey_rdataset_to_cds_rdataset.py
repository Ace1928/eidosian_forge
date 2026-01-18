import base64
import contextlib
import functools
import hashlib
import struct
import time
from datetime import datetime
from typing import Callable, Dict, List, Optional, Set, Tuple, Union, cast
import dns._features
import dns.exception
import dns.name
import dns.node
import dns.rdata
import dns.rdataclass
import dns.rdataset
import dns.rdatatype
import dns.rrset
import dns.transaction
import dns.zone
from dns.dnssectypes import Algorithm, DSDigest, NSEC3Hash
from dns.exception import (  # pylint: disable=W0611
from dns.rdtypes.ANY.CDNSKEY import CDNSKEY
from dns.rdtypes.ANY.CDS import CDS
from dns.rdtypes.ANY.DNSKEY import DNSKEY
from dns.rdtypes.ANY.DS import DS
from dns.rdtypes.ANY.NSEC import NSEC, Bitmap
from dns.rdtypes.ANY.NSEC3PARAM import NSEC3PARAM
from dns.rdtypes.ANY.RRSIG import RRSIG, sigtime_to_posixtime
from dns.rdtypes.dnskeybase import Flag
def dnskey_rdataset_to_cds_rdataset(name: Union[dns.name.Name, str], rdataset: dns.rdataset.Rdataset, algorithm: Union[DSDigest, str], origin: Optional[dns.name.Name]=None) -> dns.rdataset.Rdataset:
    """Create a CDS record from DNSKEY/CDNSKEY.

    *name*, a ``dns.name.Name`` or ``str``, the owner name of the CDS record.

    *rdataset*, a ``dns.rdataset.Rdataset``, to create DS Rdataset for.

    *algorithm*, a ``str`` or ``int`` specifying the hash algorithm.
    The currently supported hashes are "SHA1", "SHA256", and "SHA384". Case
    does not matter for these strings.

    *origin*, a ``dns.name.Name`` or ``None``.  If `key` is a relative name,
    then it will be made absolute using the specified origin.

    Raises ``UnsupportedAlgorithm`` if the algorithm is unknown or
    ``ValueError`` if the rdataset is not DNSKEY/CDNSKEY.

    Returns a ``dns.rdataset.Rdataset``
    """
    if rdataset.rdtype not in (dns.rdatatype.DNSKEY, dns.rdatatype.CDNSKEY):
        raise ValueError('rdataset not a DNSKEY/CDNSKEY')
    res = []
    for rdata in rdataset:
        res.append(make_cds(name, rdata, algorithm, origin))
    return dns.rdataset.from_rdata_list(rdataset.ttl, res)
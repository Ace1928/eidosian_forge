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
def make_ds_rdataset(rrset: Union[dns.rrset.RRset, Tuple[dns.name.Name, dns.rdataset.Rdataset]], algorithms: Set[Union[DSDigest, str]], origin: Optional[dns.name.Name]=None) -> dns.rdataset.Rdataset:
    """Create a DS record from DNSKEY/CDNSKEY/CDS.

    *rrset*, the RRset to create DS Rdataset for.  This can be a
    ``dns.rrset.RRset`` or a (``dns.name.Name``, ``dns.rdataset.Rdataset``)
    tuple.

    *algorithms*, a set of ``str`` or ``int`` specifying the hash algorithms.
    The currently supported hashes are "SHA1", "SHA256", and "SHA384". Case
    does not matter for these strings. If the RRset is a CDS, only digest
    algorithms matching algorithms are accepted.

    *origin*, a ``dns.name.Name`` or ``None``.  If `key` is a relative name,
    then it will be made absolute using the specified origin.

    Raises ``UnsupportedAlgorithm`` if any of the algorithms are unknown and
    ``ValueError`` if the given RRset is not usable.

    Returns a ``dns.rdataset.Rdataset``
    """
    rrname, rdataset = _get_rrname_rdataset(rrset)
    if rdataset.rdtype not in (dns.rdatatype.DNSKEY, dns.rdatatype.CDNSKEY, dns.rdatatype.CDS):
        raise ValueError('rrset not a DNSKEY/CDNSKEY/CDS')
    _algorithms = set()
    for algorithm in algorithms:
        try:
            if isinstance(algorithm, str):
                algorithm = DSDigest[algorithm.upper()]
        except Exception:
            raise UnsupportedAlgorithm('unsupported algorithm "%s"' % algorithm)
        _algorithms.add(algorithm)
    if rdataset.rdtype == dns.rdatatype.CDS:
        res = []
        for rdata in cds_rdataset_to_ds_rdataset(rdataset):
            if rdata.digest_type in _algorithms:
                res.append(rdata)
        if len(res) == 0:
            raise ValueError('no acceptable CDS rdata found')
        return dns.rdataset.from_rdata_list(rdataset.ttl, res)
    res = []
    for algorithm in _algorithms:
        res.extend(dnskey_rdataset_to_cds_rdataset(rrname, rdataset, algorithm, origin))
    return dns.rdataset.from_rdata_list(rdataset.ttl, res)
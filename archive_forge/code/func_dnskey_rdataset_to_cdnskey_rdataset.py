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
def dnskey_rdataset_to_cdnskey_rdataset(rdataset: dns.rdataset.Rdataset) -> dns.rdataset.Rdataset:
    """Create a CDNSKEY record from DNSKEY.

    *rdataset*, a ``dns.rdataset.Rdataset``, to create CDNSKEY Rdataset for.

    Returns a ``dns.rdataset.Rdataset``
    """
    if rdataset.rdtype != dns.rdatatype.DNSKEY:
        raise ValueError('rdataset not a DNSKEY')
    res = []
    for rdata in rdataset:
        res.append(CDNSKEY(rdclass=rdataset.rdclass, rdtype=rdataset.rdtype, flags=rdata.flags, protocol=rdata.protocol, algorithm=rdata.algorithm, key=rdata.key))
    return dns.rdataset.from_rdata_list(rdataset.ttl, res)
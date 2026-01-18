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
def sign_zone(zone: dns.zone.Zone, txn: Optional[dns.transaction.Transaction]=None, keys: Optional[List[Tuple[PrivateKey, DNSKEY]]]=None, add_dnskey: bool=True, dnskey_ttl: Optional[int]=None, inception: Optional[Union[datetime, str, int, float]]=None, expiration: Optional[Union[datetime, str, int, float]]=None, lifetime: Optional[int]=None, nsec3: Optional[NSEC3PARAM]=None, rrset_signer: Optional[RRsetSigner]=None, policy: Optional[Policy]=None) -> None:
    """Sign zone.

    *zone*, a ``dns.zone.Zone``, the zone to sign.

    *txn*, a ``dns.transaction.Transaction``, an optional transaction to use for
    signing.

    *keys*, a list of (``PrivateKey``, ``DNSKEY``) tuples, to use for signing. KSK/ZSK
    roles are assigned automatically if the SEP flag is used, otherwise all RRsets are
    signed by all keys.

    *add_dnskey*, a ``bool``.  If ``True``, the default, all specified DNSKEYs are
    automatically added to the zone on signing.

    *dnskey_ttl*, a``int``, specifies the TTL for DNSKEY RRs. If not specified the TTL
    of the existing DNSKEY RRset used or the TTL of the SOA RRset.

    *inception*, a ``datetime``, ``str``, ``int``, ``float`` or ``None``, the signature
    inception time.  If ``None``, the current time is used.  If a ``str``, the format is
    "YYYYMMDDHHMMSS" or alternatively the number of seconds since the UNIX epoch in text
    form; this is the same the RRSIG rdata's text form. Values of type `int` or `float`
    are interpreted as seconds since the UNIX epoch.

    *expiration*, a ``datetime``, ``str``, ``int``, ``float`` or ``None``, the signature
    expiration time.  If ``None``, the expiration time will be the inception time plus
    the value of the *lifetime* parameter.  See the description of *inception* above for
    how the various parameter types are interpreted.

    *lifetime*, an ``int`` or ``None``, the signature lifetime in seconds.  This
    parameter is only meaningful if *expiration* is ``None``.

    *nsec3*, a ``NSEC3PARAM`` Rdata, configures signing using NSEC3. Not yet
    implemented.

    *rrset_signer*, a ``Callable``, an optional function for signing RRsets. The
    function requires two arguments: transaction and RRset. If the not specified,
    ``dns.dnssec.default_rrset_signer`` will be used.

    Returns ``None``.
    """
    ksks = []
    zsks = []
    if keys:
        for key in keys:
            if key[1].flags & Flag.SEP:
                ksks.append(key)
            else:
                zsks.append(key)
        if not ksks:
            ksks = keys
        if not zsks:
            zsks = keys
    else:
        keys = []
    if txn:
        cm: contextlib.AbstractContextManager = contextlib.nullcontext(txn)
    else:
        cm = zone.writer()
    with cm as _txn:
        if add_dnskey:
            if dnskey_ttl is None:
                dnskey = _txn.get(zone.origin, dns.rdatatype.DNSKEY)
                if dnskey:
                    dnskey_ttl = dnskey.ttl
                else:
                    soa = _txn.get(zone.origin, dns.rdatatype.SOA)
                    dnskey_ttl = soa.ttl
            for _, dnskey in keys:
                _txn.add(zone.origin, dnskey_ttl, dnskey)
        if nsec3:
            raise NotImplementedError('Signing with NSEC3 not yet implemented')
        else:
            _rrset_signer = rrset_signer or functools.partial(default_rrset_signer, signer=zone.origin, ksks=ksks, zsks=zsks, inception=inception, expiration=expiration, lifetime=lifetime, policy=policy, origin=zone.origin)
            return _sign_zone_nsec(zone, _txn, _rrset_signer)
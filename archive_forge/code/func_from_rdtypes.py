import collections
import random
import struct
from typing import Any, List
import dns.exception
import dns.ipv4
import dns.ipv6
import dns.name
import dns.rdata
@classmethod
def from_rdtypes(cls, rdtypes: List[dns.rdatatype.RdataType]) -> 'Bitmap':
    rdtypes = sorted(rdtypes)
    window = 0
    octets = 0
    prior_rdtype = 0
    bitmap = bytearray(b'\x00' * 32)
    windows = []
    for rdtype in rdtypes:
        if rdtype == prior_rdtype:
            continue
        prior_rdtype = rdtype
        new_window = rdtype // 256
        if new_window != window:
            if octets != 0:
                windows.append((window, bytes(bitmap[0:octets])))
            bitmap = bytearray(b'\x00' * 32)
            window = new_window
        offset = rdtype % 256
        byte = offset // 8
        bit = offset % 8
        octets = byte + 1
        bitmap[byte] = bitmap[byte] | 128 >> bit
    if octets != 0:
        windows.append((window, bytes(bitmap[0:octets])))
    return cls(windows)
import collections
import datetime
import enum
import struct
import typing
from spnego._text import to_bytes, to_text
def pack_asn1_object_identifier(oid: str, tag: bool=True) -> bytes:
    """Packs an str value into an ASN.1 OBJECT IDENTIFIER byte value with optional universal tagging."""
    b_oid = bytearray()
    oid_split = [int(i) for i in oid.split('.')]
    if len(oid_split) < 2:
        raise ValueError("An OID must have 2 or more elements split by '.'")
    b_oid.append(oid_split[0] * 40 + oid_split[1])
    for val in oid_split[2:]:
        b_oid.extend(_pack_asn1_octet_number(val))
    b_value = bytes(b_oid)
    if tag:
        b_value = pack_asn1(TagClass.universal, False, TypeTagNumber.object_identifier, b_value)
    return b_value
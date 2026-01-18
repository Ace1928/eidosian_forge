import collections
import datetime
import enum
import struct
import typing
from spnego._text import to_bytes, to_text
def pack_asn1_enumerated(value: int, tag: bool=True) -> bytes:
    """Packs an int into an ASN.1 ENUMERATED byte value with optional universal tagging."""
    b_data = pack_asn1_integer(value, tag=False)
    if tag:
        b_data = pack_asn1(TagClass.universal, False, TypeTagNumber.enumerated, b_data)
    return b_data
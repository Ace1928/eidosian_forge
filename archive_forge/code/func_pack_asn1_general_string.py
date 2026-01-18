import collections
import datetime
import enum
import struct
import typing
from spnego._text import to_bytes, to_text
def pack_asn1_general_string(value: typing.Union[str, bytes], tag: bool=True, encoding: str='ascii') -> bytes:
    """Packs an string value into an ASN.1 GeneralString byte value with optional universal tagging."""
    b_data = to_bytes(value, encoding=encoding)
    if tag:
        b_data = pack_asn1(TagClass.universal, False, TypeTagNumber.general_string, b_data)
    return b_data
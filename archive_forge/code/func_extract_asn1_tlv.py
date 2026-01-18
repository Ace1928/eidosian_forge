import collections
import datetime
import enum
import struct
import typing
from spnego._text import to_bytes, to_text
def extract_asn1_tlv(tlv: typing.Union[bytes, ASN1Value], tag_class: TagClass, tag_number: typing.Union[int, TypeTagNumber]) -> bytes:
    """Extract the bytes and validates the existing tag of an ASN.1 value."""
    if isinstance(tlv, ASN1Value):
        if tag_class == TagClass.universal:
            label_name = TypeTagNumber.native_labels().get(tag_number, 'Unknown tag type')
            msg = 'Invalid ASN.1 %s tags, actual tag class %s and tag number %s' % (label_name, f'{type(tlv.tag_class).__name__}.{tlv.tag_class.name}', f'{type(tlv.tag_number).__name__}.{tlv.tag_number.name}' if isinstance(tlv.tag_number, TypeTagNumber) else tlv.tag_number)
        else:
            msg = 'Invalid ASN.1 tags, actual tag %s and number %s, expecting class %s and number %s' % (f'{type(tlv.tag_class).__name__}.{tlv.tag_class.name}', f'{type(tlv.tag_number).__name__}.{tlv.tag_number.name}' if isinstance(tlv.tag_number, TypeTagNumber) else tlv.tag_number, f'{type(tag_class).__name__}.{tag_class.name}', f'{type(tag_number).__name__}.{tag_number.name}' if isinstance(tag_number, TypeTagNumber) else tag_number)
        if tlv.tag_class != tag_class or tlv.tag_number != tag_number:
            raise ValueError(msg)
        return tlv.b_data
    return tlv
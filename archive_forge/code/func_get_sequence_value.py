import collections
import datetime
import enum
import struct
import typing
from spnego._text import to_bytes, to_text
def get_sequence_value(sequence: typing.Dict[int, ASN1Value], tag: int, structure_name: str, field_name: typing.Optional[str]=None, unpack_func: typing.Optional[typing.Callable[[typing.Union[bytes, ASN1Value]], typing.Any]]=None) -> typing.Any:
    """Gets an optional tag entry in a tagged sequence will a further unpacking of the value."""
    if tag not in sequence:
        return
    if not unpack_func:
        return sequence[tag]
    try:
        return unpack_func(sequence[tag])
    except ValueError as e:
        where = '%s in %s' % (field_name, structure_name) if field_name else structure_name
        raise ValueError('Failed unpacking %s: %s' % (where, str(e))) from e
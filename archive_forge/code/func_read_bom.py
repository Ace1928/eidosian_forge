import re
import codecs
import encodings
from typing import Callable, Match, Optional, Tuple, Union, cast
from w3lib._types import AnyUnicodeError, StrOrBytes
import w3lib.util
def read_bom(data: bytes) -> Union[Tuple[None, None], Tuple[str, bytes]]:
    """Read the byte order mark in the text, if present, and
    return the encoding represented by the BOM and the BOM.

    If no BOM can be detected, ``(None, None)`` is returned.

    >>> import w3lib.encoding
    >>> w3lib.encoding.read_bom(b'\\xfe\\xff\\x6c\\x34')
    ('utf-16-be', '\\xfe\\xff')
    >>> w3lib.encoding.read_bom(b'\\xff\\xfe\\x34\\x6c')
    ('utf-16-le', '\\xff\\xfe')
    >>> w3lib.encoding.read_bom(b'\\x00\\x00\\xfe\\xff\\x00\\x00\\x6c\\x34')
    ('utf-32-be', '\\x00\\x00\\xfe\\xff')
    >>> w3lib.encoding.read_bom(b'\\xff\\xfe\\x00\\x00\\x34\\x6c\\x00\\x00')
    ('utf-32-le', '\\xff\\xfe\\x00\\x00')
    >>> w3lib.encoding.read_bom(b'\\x01\\x02\\x03\\x04')
    (None, None)
    >>>

    """
    if data and data[0] in _FIRST_CHARS:
        for bom, encoding in _BOM_TABLE:
            if data.startswith(bom):
                return (encoding, bom)
    return (None, None)
import binascii
import codecs
import hashlib
import re
from binascii import unhexlify
from math import log10
from typing import Any, Callable, ClassVar, Dict, Optional, Sequence, Union, cast
from .._codecs import _pdfdoc_encoding_rev
from .._protocols import PdfObjectProtocol, PdfWriterProtocol
from .._utils import (
from ..errors import STREAM_TRUNCATED_PREMATURELY, PdfReadError, PdfStreamError
def encode_pdfdocencoding(unicode_string: str) -> bytes:
    retval = bytearray()
    for c in unicode_string:
        try:
            retval += b_(chr(_pdfdoc_encoding_rev[c]))
        except KeyError:
            raise UnicodeEncodeError('pdfdocencoding', c, -1, -1, 'does not exist in translation table')
    return bytes(retval)
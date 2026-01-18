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
def renumber(self) -> bytes:
    out = self[0].encode('utf-8')
    if out != b'/':
        deprecate_no_replacement(f"Incorrect first char in NameObject, should start with '/': ({self})", '6.0.0')
    for c in self[1:]:
        if c > '~':
            for x in c.encode('utf-8'):
                out += f'#{x:02X}'.encode()
        else:
            try:
                out += self.renumber_table[c]
            except KeyError:
                out += c.encode('utf-8')
    return out
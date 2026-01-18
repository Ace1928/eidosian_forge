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
def get_original_bytes(self) -> bytes:
    if self.autodetect_utf16:
        return codecs.BOM_UTF16_BE + self.encode('utf-16be')
    elif self.autodetect_pdfdocencoding:
        return encode_pdfdocencoding(self)
    else:
        raise Exception('no information about original bytes')
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
def myrepr(self) -> str:
    if self == 0:
        return '0.0'
    nb = FLOAT_WRITE_PRECISION - int(log10(abs(self)))
    s = f'{self:.{max(1, nb)}f}'.rstrip('0').rstrip('.')
    return s
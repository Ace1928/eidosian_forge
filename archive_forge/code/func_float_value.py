import io
import logging
import sys
import zlib
from typing import (
from . import settings
from .ascii85 import ascii85decode
from .ascii85 import asciihexdecode
from .ccitt import ccittfaxdecode
from .lzw import lzwdecode
from .psparser import LIT
from .psparser import PSException
from .psparser import PSObject
from .runlength import rldecode
from .utils import apply_png_predictor
def float_value(x: object) -> float:
    x = resolve1(x)
    if not isinstance(x, float):
        if settings.STRICT:
            raise PDFTypeError('Float required: %r' % x)
        return 0.0
    return x
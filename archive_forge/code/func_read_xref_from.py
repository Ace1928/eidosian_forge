import itertools
import logging
import re
import struct
from hashlib import sha256, md5, sha384, sha512
from typing import (
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from . import settings
from .arcfour import Arcfour
from .data_structures import NumberTree
from .pdfparser import PDFSyntaxError, PDFParser, PDFStreamParser
from .pdftypes import (
from .psparser import PSEOF, literal_name, LIT, KWD
from .utils import choplist, decode_text, nunpack, format_int_roman, format_int_alpha
def read_xref_from(self, parser: PDFParser, start: int, xrefs: List[PDFBaseXRef]) -> None:
    """Reads XRefs from the given location."""
    parser.seek(start)
    parser.reset()
    try:
        pos, token = parser.nexttoken()
    except PSEOF:
        raise PDFNoValidXRef('Unexpected EOF')
    log.debug('read_xref_from: start=%d, token=%r', start, token)
    if isinstance(token, int):
        parser.seek(pos)
        parser.reset()
        xref: PDFBaseXRef = PDFXRefStream()
        xref.load(parser)
    else:
        if token is parser.KEYWORD_XREF:
            parser.nextline()
        xref = PDFXRef()
        xref.load(parser)
    xrefs.append(xref)
    trailer = xref.get_trailer()
    log.debug('trailer: %r', trailer)
    if 'XRefStm' in trailer:
        pos = int_value(trailer['XRefStm'])
        self.read_xref_from(parser, pos, xrefs)
    if 'Prev' in trailer:
        pos = int_value(trailer['Prev'])
        self.read_xref_from(parser, pos, xrefs)
    return
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
def compute_encryption_key(self, password: bytes) -> bytes:
    password = (password + self.PASSWORD_PADDING)[:32]
    hash = md5(password)
    hash.update(self.o)
    hash.update(struct.pack('<L', self.p))
    hash.update(self.docid[0])
    if self.r >= 4:
        if not cast(PDFStandardSecurityHandlerV4, self).encrypt_metadata:
            hash.update(b'\xff\xff\xff\xff')
    result = hash.digest()
    n = 5
    if self.r >= 3:
        n = self.length // 8
        for _ in range(50):
            result = md5(result[:n]).digest()
    return result[:n]
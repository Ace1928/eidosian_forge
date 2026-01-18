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
def authenticate_owner_password(self, password: bytes) -> Optional[bytes]:
    password = (password + self.PASSWORD_PADDING)[:32]
    hash = md5(password)
    if self.r >= 3:
        for _ in range(50):
            hash = md5(hash.digest())
    n = 5
    if self.r >= 3:
        n = self.length // 8
    key = hash.digest()[:n]
    if self.r == 2:
        user_password = Arcfour(key).decrypt(self.o)
    else:
        user_password = self.o
        for i in range(19, -1, -1):
            k = b''.join((bytes((c ^ i,)) for c in iter(key)))
            user_password = Arcfour(k).decrypt(user_password)
    return self.authenticate_user_password(user_password)
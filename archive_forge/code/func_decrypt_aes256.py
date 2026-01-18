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
def decrypt_aes256(self, objid: int, genno: int, data: bytes) -> bytes:
    initialization_vector = data[:16]
    ciphertext = data[16:]
    assert self.key is not None
    cipher = Cipher(algorithms.AES(self.key), modes.CBC(initialization_vector), backend=default_backend())
    return cipher.decryptor().update(ciphertext)
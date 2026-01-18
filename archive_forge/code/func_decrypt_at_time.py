from __future__ import annotations
import base64
import binascii
import os
import time
import typing
from cryptography import utils
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.hmac import HMAC
def decrypt_at_time(self, msg: typing.Union[bytes, str], ttl: int, current_time: int) -> bytes:
    for f in self._fernets:
        try:
            return f.decrypt_at_time(msg, ttl, current_time)
        except InvalidToken:
            pass
    raise InvalidToken
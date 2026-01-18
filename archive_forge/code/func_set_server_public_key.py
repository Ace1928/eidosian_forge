import hmac
import math
import os
from hashlib import sha256
from typing import Optional  # Needed for mypy
def set_server_public_key(self, server_public_key: int) -> None:
    common_secret_int = pow(server_public_key, self.my_private_key, DH_PRIME_1024)
    common_secret = int_to_bytes(common_secret_int)
    common_secret = b'\x00' * (128 - len(common_secret)) + common_secret
    salt = b'\x00' * 32
    pseudo_random_key = hmac.new(salt, common_secret, sha256).digest()
    output_block = hmac.new(pseudo_random_key, b'\x01', sha256).digest()
    self.aes_key = output_block[:16]
import secrets
from Crypto import __version__
from Crypto.Cipher import AES, ARC4
from Crypto.Util.Padding import pad
from pypdf._crypt_providers._base import CryptBase
def rc4_encrypt(key: bytes, data: bytes) -> bytes:
    return ARC4.ARC4Cipher(key).encrypt(data)
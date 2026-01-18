import secrets
from Crypto import __version__
from Crypto.Cipher import AES, ARC4
from Crypto.Util.Padding import pad
from pypdf._crypt_providers._base import CryptBase
def aes_ecb_decrypt(key: bytes, data: bytes) -> bytes:
    return AES.new(key, AES.MODE_ECB).decrypt(data)
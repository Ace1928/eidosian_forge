from .err import OperationalError
from functools import partial
import hashlib
def sha2_rsa_encrypt(password, salt, public_key):
    """Encrypt password with salt and public_key.

    Used for sha256_password and caching_sha2_password.
    """
    if not _have_cryptography:
        raise RuntimeError("'cryptography' package is required for sha256_password or" + ' caching_sha2_password auth methods')
    message = _xor_password(password + b'\x00', salt)
    rsa_key = serialization.load_pem_public_key(public_key, default_backend())
    return rsa_key.encrypt(message, padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA1()), algorithm=hashes.SHA1(), label=None))
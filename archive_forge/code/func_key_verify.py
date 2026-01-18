import cryptography.hazmat.primitives.asymmetric as _asymmetric
import cryptography.hazmat.primitives.hashes as _hashes
import cryptography.hazmat.primitives.serialization as _serialization
def key_verify(rsakey, signature, message, digest):
    """Verify the given signature with the RSA key."""
    padding = _asymmetric.padding.PKCS1v15()
    if isinstance(rsakey, _asymmetric.rsa.RSAPrivateKey):
        rsakey = rsakey.public_key()
    try:
        rsakey.verify(signature, message, padding, digest)
    except Exception:
        return False
    else:
        return True
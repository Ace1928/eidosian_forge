import cryptography.hazmat.primitives.asymmetric as _asymmetric
import cryptography.hazmat.primitives.hashes as _hashes
import cryptography.hazmat.primitives.serialization as _serialization
def key_sign(rsakey, message, digest):
    """Sign the given message with the RSA key."""
    padding = _asymmetric.padding.PKCS1v15()
    signature = rsakey.sign(message, padding, digest)
    return signature
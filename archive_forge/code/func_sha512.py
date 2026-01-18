import nacl.bindings
import nacl.encoding
def sha512(message: bytes, encoder: nacl.encoding.Encoder=nacl.encoding.HexEncoder) -> bytes:
    """
    Hashes ``message`` with SHA512.

    :param message: The message to hash.
    :type message: bytes
    :param encoder: A class that is able to encode the hashed message.
    :returns: The hashed message.
    :rtype: bytes
    """
    return encoder.encode(nacl.bindings.crypto_hash_sha512(message))
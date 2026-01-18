import nacl.bindings
import nacl.encoding
def siphashx24(message: bytes, key: bytes=b'', encoder: nacl.encoding.Encoder=nacl.encoding.HexEncoder) -> bytes:
    """
    Computes a keyed MAC of ``message`` using the 128 bit variant of the
    siphash-2-4 construction.

    :param message: The message to hash.
    :type message: bytes
    :param key: the message authentication key for the siphash MAC construct
    :type key: bytes(:const:`SIPHASHX_KEYBYTES`)
    :param encoder: A class that is able to encode the hashed message.
    :returns: The hashed message.
    :rtype: bytes(:const:`SIPHASHX_BYTES`)
    :raises nacl.exceptions.UnavailableError: If called when using a
        minimal build of libsodium.

    .. versionadded:: 1.2
    """
    digest = _sip_hashx(message, key)
    return encoder.encode(digest)
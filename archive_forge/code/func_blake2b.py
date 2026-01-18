import nacl.bindings
import nacl.encoding
def blake2b(data: bytes, digest_size: int=BLAKE2B_BYTES, key: bytes=b'', salt: bytes=b'', person: bytes=b'', encoder: nacl.encoding.Encoder=nacl.encoding.HexEncoder) -> bytes:
    """
    Hashes ``data`` with blake2b.

    :param data: the digest input byte sequence
    :type data: bytes
    :param digest_size: the requested digest size; must be at most
                        :const:`BLAKE2B_BYTES_MAX`;
                        the default digest size is
                        :const:`BLAKE2B_BYTES`
    :type digest_size: int
    :param key: the key to be set for keyed MAC/PRF usage; if set, the key
                must be at most :data:`~nacl.hash.BLAKE2B_KEYBYTES_MAX` long
    :type key: bytes
    :param salt: an initialization salt at most
                 :const:`BLAKE2B_SALTBYTES` long;
                 it will be zero-padded if needed
    :type salt: bytes
    :param person: a personalization string at most
                   :const:`BLAKE2B_PERSONALBYTES` long;
                   it will be zero-padded if needed
    :type person: bytes
    :param encoder: the encoder to use on returned digest
    :type encoder: class
    :returns: The hashed message.
    :rtype: bytes
    """
    digest = _b2b_hash(data, digest_size=digest_size, key=key, salt=salt, person=person)
    return encoder.encode(digest)
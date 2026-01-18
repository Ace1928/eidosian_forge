from itertools import chain
import struct
from passlib.utils import getrandbytes, rng
from passlib.utils.binary import bcrypt64
from passlib.utils.compat import BytesIO, unicode, u, native_string_types
from passlib.crypto._blowfish.unrolled import BlowfishEngine
def raw_bcrypt(password, ident, salt, log_rounds):
    """perform central password hashing step in bcrypt scheme.

    :param password: the password to hash
    :param ident: identifier w/ minor version (e.g. 2, 2a)
    :param salt: the binary salt to use (encoded in bcrypt-base64)
    :param log_rounds: the log2 of the number of rounds (as int)
    :returns: bcrypt-base64 encoded checksum
    """
    assert isinstance(ident, native_string_types)
    add_null_padding = True
    if ident == u('2a') or ident == u('2y') or ident == u('2b'):
        pass
    elif ident == u('2'):
        add_null_padding = False
    elif ident == u('2x'):
        raise ValueError("crypt_blowfish's buggy '2x' hashes are not currently supported")
    else:
        raise ValueError('unknown ident: %r' % (ident,))
    assert isinstance(salt, bytes)
    salt = bcrypt64.decode_bytes(salt)
    if len(salt) < 16:
        raise ValueError('Missing salt bytes')
    elif len(salt) > 16:
        salt = salt[:16]
    assert isinstance(password, bytes)
    if add_null_padding:
        password += BNULL
    if log_rounds < 4 or log_rounds > 31:
        raise ValueError('Bad number of rounds')
    engine = BlowfishEngine()
    pass_words = engine.key_to_words(password)
    salt_words = engine.key_to_words(salt)
    salt_words16 = salt_words[:4]
    engine.eks_salted_expand(pass_words, salt_words16)
    rounds = 1 << log_rounds
    engine.eks_repeated_expand(pass_words, salt_words, rounds)
    data = list(BCRYPT_CDATA)
    i = 0
    while i < 6:
        data[i], data[i + 1] = engine.repeat_encipher(data[i], data[i + 1], 64)
        i += 2
    raw = digest_struct.pack(*data)[:-1]
    return bcrypt64.encode_bytes(raw)
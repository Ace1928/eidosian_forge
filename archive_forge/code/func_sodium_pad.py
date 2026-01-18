import nacl.exceptions as exc
from nacl._sodium import ffi, lib
from nacl.exceptions import ensure
def sodium_pad(s: bytes, blocksize: int) -> bytes:
    """
    Pad the input bytearray ``s`` to a multiple of ``blocksize``
    using the ISO/IEC 7816-4 algorithm

    :param s: input bytes string
    :type s: bytes
    :param blocksize:
    :type blocksize: int
    :return: padded string
    :rtype: bytes
    """
    ensure(isinstance(s, bytes), raising=exc.TypeError)
    ensure(isinstance(blocksize, int), raising=exc.TypeError)
    if blocksize <= 0:
        raise exc.ValueError
    s_len = len(s)
    m_len = s_len + blocksize
    buf = ffi.new('unsigned char []', m_len)
    p_len = ffi.new('size_t []', 1)
    ffi.memmove(buf, s, s_len)
    rc = lib.sodium_pad(p_len, buf, s_len, blocksize, m_len)
    ensure(rc == 0, 'Padding failure', raising=exc.CryptoError)
    return ffi.buffer(buf, p_len[0])[:]
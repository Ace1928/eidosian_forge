import nacl.exceptions as exc
from nacl._sodium import ffi, lib
from nacl.exceptions import ensure
def sodium_add(a: bytes, b: bytes) -> bytes:
    """
    Given a couple of *same-sized* byte sequences, interpreted as the
    little-endian representation of two unsigned integers, compute
    the modular addition of the represented values, in constant time for
    a given common length of the byte sequences.

    :param a: input bytes buffer
    :type a: bytes
    :param b: input bytes buffer
    :type b: bytes
    :return: a byte-sequence representing, as a little-endian big integer,
             the integer value of ``(to_int(a) + to_int(b)) mod 2^(8*len(a))``
    :rtype: bytes
    """
    ensure(isinstance(a, bytes), raising=exc.TypeError)
    ensure(isinstance(b, bytes), raising=exc.TypeError)
    ln = len(a)
    ensure(len(b) == ln, raising=exc.TypeError)
    buf_a = ffi.new('unsigned char []', ln)
    buf_b = ffi.new('unsigned char []', ln)
    ffi.memmove(buf_a, a, ln)
    ffi.memmove(buf_b, b, ln)
    lib.sodium_add(buf_a, buf_b, ln)
    return ffi.buffer(buf_a, ln)[:]
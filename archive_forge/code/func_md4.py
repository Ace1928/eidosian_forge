import struct
import typing as t
def md4(data: bytes) -> bytes:
    """Python implementation of md4 hashing.

    This is a pure Python implementation of the `MD4 Hashing Algorithm`_.
    Recent distributions of Linux ship with OpenSSL 3.x which has disabled the
    md4 engine so for backwards compatibility this must be manually implemented
    here to ensure NTLM continues to work.

    .. _MD4 Hasing Algorithm:
        https://datatracker.ietf.org/doc/html/rfc1320
    """
    data_block = bytearray(data)
    orig_length = len(data)
    b_orig_length = (orig_length * 8 & 18446744073709551615).to_bytes(8, 'little')
    data_block += b''.join([b'\x80', bytes((55 - orig_length % 64) % 64), b_orig_length])
    A = 1732584193
    B = 4023233417
    C = 2562383102
    D = 271733878
    while data_block:
        AA, BB, CC, DD = (A, B, C, D)
        X = struct.unpack('<16I', data_block[:64])
        data_block = data_block[64:]
        AA = FF(AA, BB, CC, DD, X[0], 3)
        DD = FF(DD, AA, BB, CC, X[1], 7)
        CC = FF(CC, DD, AA, BB, X[2], 11)
        BB = FF(BB, CC, DD, AA, X[3], 19)
        AA = FF(AA, BB, CC, DD, X[4], 3)
        DD = FF(DD, AA, BB, CC, X[5], 7)
        CC = FF(CC, DD, AA, BB, X[6], 11)
        BB = FF(BB, CC, DD, AA, X[7], 19)
        AA = FF(AA, BB, CC, DD, X[8], 3)
        DD = FF(DD, AA, BB, CC, X[9], 7)
        CC = FF(CC, DD, AA, BB, X[10], 11)
        BB = FF(BB, CC, DD, AA, X[11], 19)
        AA = FF(AA, BB, CC, DD, X[12], 3)
        DD = FF(DD, AA, BB, CC, X[13], 7)
        CC = FF(CC, DD, AA, BB, X[14], 11)
        BB = FF(BB, CC, DD, AA, X[15], 19)
        AA = GG(AA, BB, CC, DD, X[0], 3)
        DD = GG(DD, AA, BB, CC, X[4], 5)
        CC = GG(CC, DD, AA, BB, X[8], 9)
        BB = GG(BB, CC, DD, AA, X[12], 13)
        AA = GG(AA, BB, CC, DD, X[1], 3)
        DD = GG(DD, AA, BB, CC, X[5], 5)
        CC = GG(CC, DD, AA, BB, X[9], 9)
        BB = GG(BB, CC, DD, AA, X[13], 13)
        AA = GG(AA, BB, CC, DD, X[2], 3)
        DD = GG(DD, AA, BB, CC, X[6], 5)
        CC = GG(CC, DD, AA, BB, X[10], 9)
        BB = GG(BB, CC, DD, AA, X[14], 13)
        AA = GG(AA, BB, CC, DD, X[3], 3)
        DD = GG(DD, AA, BB, CC, X[7], 5)
        CC = GG(CC, DD, AA, BB, X[11], 9)
        BB = GG(BB, CC, DD, AA, X[15], 13)
        AA = HH(AA, BB, CC, DD, X[0], 3)
        DD = HH(DD, AA, BB, CC, X[8], 9)
        CC = HH(CC, DD, AA, BB, X[4], 11)
        BB = HH(BB, CC, DD, AA, X[12], 15)
        AA = HH(AA, BB, CC, DD, X[2], 3)
        DD = HH(DD, AA, BB, CC, X[10], 9)
        CC = HH(CC, DD, AA, BB, X[6], 11)
        BB = HH(BB, CC, DD, AA, X[14], 15)
        AA = HH(AA, BB, CC, DD, X[1], 3)
        DD = HH(DD, AA, BB, CC, X[9], 9)
        CC = HH(CC, DD, AA, BB, X[5], 11)
        BB = HH(BB, CC, DD, AA, X[13], 15)
        AA = HH(AA, BB, CC, DD, X[3], 3)
        DD = HH(DD, AA, BB, CC, X[11], 9)
        CC = HH(CC, DD, AA, BB, X[7], 11)
        BB = HH(BB, CC, DD, AA, X[15], 15)
        A = A + AA & 4294967295
        B = B + BB & 4294967295
        C = C + CC & 4294967295
        D = D + DD & 4294967295
    return struct.pack('<IIII', A, B, C, D)
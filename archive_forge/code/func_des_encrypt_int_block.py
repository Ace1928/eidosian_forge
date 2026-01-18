import struct
from passlib import exc
from passlib.utils.compat import join_byte_values, byte_elem_value, \
def des_encrypt_int_block(key, input, salt=0, rounds=1):
    """encrypt single block of data using DES, operates on 64-bit integers.

    this function is essentially the same as :func:`des_encrypt_block`,
    except that it operates on integers, and will NOT automatically
    expand 56-bit keys if provided (since there's no way to detect them).

    :arg key:
        DES key as 64-bit integer (the parity bits are ignored).

    :arg input:
        input block as 64-bit integer

    :arg salt:
        optional 24-bit integer used to mutate the base DES algorithm.
        defaults to ``0`` (no mutation applied).

    :arg rounds:
        optional number of rounds of to apply the DES key schedule.
        defaults to ``1``.

    :raises TypeError: if any of the provided args are of the wrong type.
    :raises ValueError:
        if any of the input blocks are the wrong size,
        or the salt/rounds values are out of range.

    :returns:
        resulting ciphertext as 64-bit integer.
    """
    if rounds < 1:
        raise ValueError('rounds must be positive integer')
    if salt < 0 or salt > INT_24_MASK:
        raise ValueError('salt must be 24-bit non-negative integer')
    if not isinstance(key, int_types):
        raise exc.ExpectedTypeError(key, 'int', 'key')
    elif key < 0 or key > INT_64_MASK:
        raise ValueError('key must be 64-bit non-negative integer')
    if not isinstance(input, int_types):
        raise exc.ExpectedTypeError(input, 'int', 'input')
    elif input < 0 or input > INT_64_MASK:
        raise ValueError('input must be 64-bit non-negative integer')
    global SPE, PCXROT, IE3264, CF6464
    if PCXROT is None:
        _load_tables()
    SPE0, SPE1, SPE2, SPE3, SPE4, SPE5, SPE6, SPE7 = SPE

    def _iter_key_schedule(ks_odd):
        """given 64-bit key, iterates over the 8 (even,odd) key schedule pairs"""
        for p_even, p_odd in PCXROT:
            ks_even = _permute(ks_odd, p_even)
            ks_odd = _permute(ks_even, p_odd)
            yield (ks_even & _KS_MASK, ks_odd & _KS_MASK)
    ks_list = list(_iter_key_schedule(key))
    salt = (salt & 63) << 26 | (salt & 4032) << 12 | (salt & 258048) >> 2 | (salt & 16515072) >> 16
    if input == 0:
        L = R = 0
    else:
        L = input >> 31 & 2863311530 | input & 1431655765
        L = _permute(L, IE3264)
        R = input >> 32 & 2863311530 | input >> 1 & 1431655765
        R = _permute(R, IE3264)
    while rounds:
        rounds -= 1
        for ks_even, ks_odd in ks_list:
            k = (R >> 32 ^ R) & salt
            B = k << 32 ^ k ^ R ^ ks_even
            L ^= SPE0[B >> 58 & 63] ^ SPE1[B >> 50 & 63] ^ SPE2[B >> 42 & 63] ^ SPE3[B >> 34 & 63] ^ SPE4[B >> 26 & 63] ^ SPE5[B >> 18 & 63] ^ SPE6[B >> 10 & 63] ^ SPE7[B >> 2 & 63]
            k = (L >> 32 ^ L) & salt
            B = k << 32 ^ k ^ L ^ ks_odd
            R ^= SPE0[B >> 58 & 63] ^ SPE1[B >> 50 & 63] ^ SPE2[B >> 42 & 63] ^ SPE3[B >> 34 & 63] ^ SPE4[B >> 26 & 63] ^ SPE5[B >> 18 & 63] ^ SPE6[B >> 10 & 63] ^ SPE7[B >> 2 & 63]
        L, R = (R, L)
    C = L >> 3 & 1085102592318504960 | L << 33 & 17361641477096079360 | R >> 35 & 252645135 | R << 1 & 4042322160
    return _permute(C, CF6464)
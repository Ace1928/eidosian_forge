from sympy.core import Basic, Integer
import random
def gray_to_bin(bin_list):
    """
    Convert from Gray coding to binary coding.

    We assume big endian encoding.

    Examples
    ========

    >>> from sympy.combinatorics.graycode import gray_to_bin
    >>> gray_to_bin('100')
    '111'

    See Also
    ========

    bin_to_gray
    """
    b = [bin_list[0]]
    for i in range(1, len(bin_list)):
        b += str(int(b[i - 1] != bin_list[i]))
    return ''.join(b)
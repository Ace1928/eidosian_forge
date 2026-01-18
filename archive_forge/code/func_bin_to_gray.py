from sympy.core import Basic, Integer
import random
def bin_to_gray(bin_list):
    """
    Convert from binary coding to gray coding.

    We assume big endian encoding.

    Examples
    ========

    >>> from sympy.combinatorics.graycode import bin_to_gray
    >>> bin_to_gray('111')
    '100'

    See Also
    ========

    gray_to_bin
    """
    b = [bin_list[0]]
    for i in range(1, len(bin_list)):
        b += str(int(bin_list[i]) ^ int(bin_list[i - 1]))
    return ''.join(b)
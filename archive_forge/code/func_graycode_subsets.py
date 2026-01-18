from sympy.core import Basic, Integer
import random
def graycode_subsets(gray_code_set):
    """
    Generates the subsets as enumerated by a Gray code.

    Examples
    ========

    >>> from sympy.combinatorics.graycode import graycode_subsets
    >>> list(graycode_subsets(['a', 'b', 'c']))
    [[], ['c'], ['b', 'c'], ['b'], ['a', 'b'], ['a', 'b', 'c'],     ['a', 'c'], ['a']]
    >>> list(graycode_subsets(['a', 'b', 'c', 'c']))
    [[], ['c'], ['c', 'c'], ['c'], ['b', 'c'], ['b', 'c', 'c'],     ['b', 'c'], ['b'], ['a', 'b'], ['a', 'b', 'c'], ['a', 'b', 'c', 'c'],     ['a', 'b', 'c'], ['a', 'c'], ['a', 'c', 'c'], ['a', 'c'], ['a']]

    See Also
    ========

    get_subset_from_bitstring
    """
    for bitstring in list(GrayCode(len(gray_code_set)).generate_gray()):
        yield get_subset_from_bitstring(gray_code_set, bitstring)
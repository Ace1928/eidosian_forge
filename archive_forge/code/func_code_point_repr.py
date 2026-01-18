from sys import maxunicode
from typing import Iterable, Iterator, Optional, Set, Tuple, Union
def code_point_repr(cp: CodePoint) -> str:
    """
    Returns the string representation of a code point.

    :param cp: an integer or a tuple with at least two integers.     Values must be in interval [0, sys.maxunicode].
    """
    if isinstance(cp, int):
        if cp in CHARACTER_CLASS_ESCAPED:
            return '\\%s' % chr(cp)
        return chr(cp)
    if cp[0] in CHARACTER_CLASS_ESCAPED:
        start_char = '\\%s' % chr(cp[0])
    else:
        start_char = chr(cp[0])
    end_cp = cp[1] - 1
    if end_cp in CHARACTER_CLASS_ESCAPED:
        end_char = '\\%s' % chr(end_cp)
    else:
        end_char = chr(end_cp)
    if end_cp > cp[0] + 1:
        return '%s-%s' % (start_char, end_char)
    else:
        return start_char + end_char
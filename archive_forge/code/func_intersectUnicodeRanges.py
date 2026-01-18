from fontTools.misc import sstruct
from fontTools.misc.roundTools import otRound
from fontTools.misc.textTools import safeEval, num2binary, binary2num
from fontTools.ttLib.tables import DefaultTable
import bisect
import logging
def intersectUnicodeRanges(unicodes, inverse=False):
    """Intersect a sequence of (int) Unicode codepoints with the Unicode block
    ranges defined in the OpenType specification v1.7, and return the set of
    'ulUnicodeRanges' bits for which there is at least ONE intersection.
    If 'inverse' is True, return the the bits for which there is NO intersection.

    >>> intersectUnicodeRanges([0x0410]) == {9}
    True
    >>> intersectUnicodeRanges([0x0410, 0x1F000]) == {9, 57, 122}
    True
    >>> intersectUnicodeRanges([0x0410, 0x1F000], inverse=True) == (
    ...     set(range(len(OS2_UNICODE_RANGES))) - {9, 57, 122})
    True
    """
    unicodes = set(unicodes)
    unicodestarts, unicodevalues = _getUnicodeRanges()
    bits = set()
    for code in unicodes:
        stop, bit = unicodevalues[bisect.bisect(unicodestarts, code)]
        if code <= stop:
            bits.add(bit)
    if any((65536 <= code < 1114112 for code in unicodes)):
        bits.add(57)
    return set(range(len(OS2_UNICODE_RANGES))) - bits if inverse else bits
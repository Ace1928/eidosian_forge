from __future__ import annotations
import re
from fractions import Fraction
def str_delimited(results, header=None, delimiter='\t'):
    """Given a tuple of tuples, generate a delimited string form.
    >>> results = [["a","b","c"],["d","e","f"],[1,2,3]]
    >>> print(str_delimited(results,delimiter=","))
    a,b,c
    d,e,f
    1,2,3.

    Args:
        results: 2d sequence of arbitrary types.
        header: optional header
        delimiter: Defaults to "\\t" for tab-delimited output.

    Returns:
        Aligned string output in a table-like format.
    """
    out = ''
    if header is not None:
        out += delimiter.join(header) + '\n'
    return out + '\n'.join((delimiter.join([str(m) for m in result]) for result in results))
import decimal
import json as _json
import sys
import re
from functools import reduce
from _plotly_utils.optional_imports import get_module
from _plotly_utils.basevalidators import ImageUriValidator
def split_string_positions(ss):
    """
    Given a list of strings split using split_multichar, return a list of
    integers representing the indices of the first character of every string in
    the original string.
    Example:

        >>> ss = ["a.string[0].with_separators"]
        >>> chars = list(".[]_")
        >>> ss_split = split_multichar(ss, chars)
        >>> ss_split
        ['a', 'string', '0', '', 'with', 'separators']
        >>> split_string_positions(ss_split)
        [0, 2, 9, 11, 12, 17]

    :param (list) ss: A list of strings.
    """
    return list(map(lambda t: t[0] + t[1], zip(range(len(ss)), cumsum([0] + list(map(len, ss[:-1]))))))
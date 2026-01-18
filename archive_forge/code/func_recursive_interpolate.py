import os
import re
import sys
from codecs import BOM_UTF8, BOM_UTF16, BOM_UTF16_BE, BOM_UTF16_LE
import six
from ._version import __version__
def recursive_interpolate(key, value, section, backtrail):
    """The function that does the actual work.

            ``value``: the string we're trying to interpolate.
            ``section``: the section in which that string was found
            ``backtrail``: a dict to keep track of where we've been,
            to detect and prevent infinite recursion loops

            This is similar to a depth-first-search algorithm.
            """
    if (key, section.name) in backtrail:
        raise InterpolationLoopError(key)
    backtrail[key, section.name] = 1
    match = self._KEYCRE.search(value)
    while match:
        k, v, s = self._parse_match(match)
        if k is None:
            replacement = v
        else:
            replacement = recursive_interpolate(k, v, s, backtrail)
        start, end = match.span()
        value = ''.join((value[:start], replacement, value[end:]))
        new_search_start = start + len(replacement)
        match = self._KEYCRE.search(value, new_search_start)
    del backtrail[key, section.name]
    return value
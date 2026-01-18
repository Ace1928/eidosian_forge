from collections import namedtuple
import logging
import re
from ._mathtext_data import uni2type1
def string_width_height(self, s):
    """
        Return the string width (including kerning) and string height
        as a (*w*, *h*) tuple.
        """
    if not len(s):
        return (0, 0)
    total_width = 0
    namelast = None
    miny = 1000000000.0
    maxy = 0
    for c in s:
        if c == '\n':
            continue
        wx, name, bbox = self._metrics[ord(c)]
        total_width += wx + self._kern.get((namelast, name), 0)
        l, b, w, h = bbox
        miny = min(miny, b)
        maxy = max(maxy, b + h)
        namelast = name
    return (total_width, maxy - miny)
import re
from kivy.properties import dpi2px
from kivy.parser import parse_color
from kivy.logger import Logger
from kivy.core.text import Label, LabelBase
from kivy.core.text.text_layout import layout_text, LayoutWord, LayoutLine
from copy import copy
from functools import partial
def n_restricted(line, uw, c):
    """ Similar to the function `n`, except it only returns the first
            occurrence and it's not an iterator. Furthermore, if the first
            occurrence doesn't fit within width uw, it returns the index of
            whatever amount of text will still fit in uw.

            :returns:
                similar to the function `n`, except it's a 4-tuple, with the
                last element a boolean, indicating if we had to clip the text
                to fit in uw (True) or if the whole text until the first
                occurrence fitted in uw (False).
            """
    total_w = 0
    if not len(line):
        return (0, 0, 0)
    for w in range(len(line)):
        word = line[w]
        f = partial(word.text.find, c)
        self.options = word.options
        extents = self.get_cached_extents()
        i = f()
        if i != -1:
            ww = extents(word.text[:i])[0]
        if i != -1 and total_w + ww <= uw:
            return (w, i, total_w + ww, False)
        elif i == -1:
            ww = extents(word.text)[0]
            if total_w + ww <= uw:
                total_w += ww
                continue
            i = len(word.text)
        e = 0
        while e != i and total_w + extents(word.text[:e])[0] <= uw:
            e += 1
        e = max(0, e - 1)
        return (w, e, total_w + extents(word.text[:e])[0], True)
    return (-1, -1, total_w, False)
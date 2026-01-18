import re
from kivy.properties import dpi2px
from kivy.parser import parse_color
from kivy.logger import Logger
from kivy.core.text import Label, LabelBase
from kivy.core.text.text_layout import layout_text, LayoutWord, LayoutLine
from copy import copy
from functools import partial
def p_restricted(line, uw, c):
    """ Similar to `n_restricted`, except it returns the first
            occurrence starting from the right, like `p`.
            """
    total_w = 0
    if not len(line):
        return (0, 0, 0)
    for w in range(len(line) - 1, -1, -1):
        word = line[w]
        f = partial(word.text.rfind, c)
        self.options = word.options
        extents = self.get_cached_extents()
        i = f()
        if i != -1:
            ww = extents(word.text[i + 1:])[0]
        if i != -1 and total_w + ww <= uw:
            return (w, i, total_w + ww, False)
        elif i == -1:
            ww = extents(word.text)[0]
            if total_w + ww <= uw:
                total_w += ww
                continue
        s = len(word.text) - 1
        while s >= 0 and total_w + extents(word.text[s:])[0] <= uw:
            s -= 1
        return (w, s, total_w + extents(word.text[s + 1:])[0], True)
    return (-1, -1, total_w, False)
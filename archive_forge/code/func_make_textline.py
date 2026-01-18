import argparse
import bisect
import os
import sys
import statistics
from typing import Dict, List, Set
from . import fitz
def make_textline(left, slot, minslot, lchars):
    """Produce the text of one output line.

        Args:
            left: (float) left most coordinate used on page
            slot: (float) avg width of one character in any font in use.
            minslot: (float) min width for the characters in this line.
            chars: (list[tuple]) characters of this line.
        Returns:
            text: (str) text string for this line
        """
    text = ''
    old_char = ''
    old_x1 = 0
    old_ox = 0
    if minslot <= fitz.EPSILON:
        raise RuntimeError('program error: minslot too small = %g' % minslot)
    for c in lchars:
        char, ox, _, cwidth = c
        ox = ox - left
        x1 = ox + cwidth
        if old_char == char and ox - old_ox <= cwidth * 0.2:
            continue
        if char == ' ' and (old_x1 - ox) / cwidth > 0.8:
            continue
        old_char = char
        if ox < old_x1 + minslot:
            text += char
            old_x1 = x1
            old_ox = ox
            continue
        if char == ' ':
            continue
        delta = int(ox / slot) - len(text)
        if ox > old_x1 and delta > 1:
            text += ' ' * delta
        text += char
        old_x1 = x1
        old_ox = ox
    return text.rstrip()
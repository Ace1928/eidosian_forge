import tkinter
from tkinter.constants import *
def parse_dash(string, width):
    """parse dash pattern specified with string"""
    w = max(1, int(width + 0.5))
    n = len(string)
    result = []
    for i, c in enumerate(string):
        if c == ' ' and len(result):
            result[-1] += w + 1
        elif c == '_':
            result.append(8 * w)
            result.append(4 * w)
        elif c == '-':
            result.append(6 * w)
            result.append(4 * w)
        elif c == ',':
            result.append(4 * w)
            result.append(4 * w)
        elif c == '.':
            result.append(2 * w)
            result.append(4 * w)
    return result
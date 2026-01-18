import unicodedata
from pyglet.gl import *
from pyglet import image
def grapheme_break(left, right):
    if left is None:
        return True
    if left == _CR and right == _LF:
        return False
    left_cc = unicodedata.category(left)
    if _control(left, left_cc):
        return True
    right_cc = unicodedata.category(right)
    if _control(right, right_cc):
        return True
    if _extend(right, right_cc):
        return False
    if _spacing_mark(right, right_cc):
        return False
    if _prepend(left, left_cc):
        return False
    return True
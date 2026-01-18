import inspect
import itertools
import string
import html
from collections.abc import Sequence
from dataclasses import dataclass
from operator import itemgetter
from . import (
def get_bbox_overlap(a, b):
    a_left, a_top, a_right, a_bottom = a
    b_left, b_top, b_right, b_bottom = b
    o_left = max(a_left, b_left)
    o_right = min(a_right, b_right)
    o_bottom = min(a_bottom, b_bottom)
    o_top = max(a_top, b_top)
    o_width = o_right - o_left
    o_height = o_bottom - o_top
    if o_height >= 0 and o_width >= 0 and (o_height + o_width > 0):
        return (o_left, o_top, o_right, o_bottom)
    else:
        return None
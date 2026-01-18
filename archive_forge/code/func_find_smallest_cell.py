import inspect
import itertools
import string
import html
from collections.abc import Sequence
from dataclasses import dataclass
from operator import itemgetter
from . import (
def find_smallest_cell(points, i: int):
    if i == n_points - 1:
        return None
    pt = points[i]
    rest = points[i + 1:]
    below = [x for x in rest if x[0] == pt[0]]
    right = [x for x in rest if x[1] == pt[1]]
    for below_pt in below:
        if not edge_connects(pt, below_pt):
            continue
        for right_pt in right:
            if not edge_connects(pt, right_pt):
                continue
            bottom_right = (right_pt[0], below_pt[1])
            if bottom_right in intersections and edge_connects(bottom_right, right_pt) and edge_connects(bottom_right, below_pt):
                return (pt[0], pt[1], bottom_right[0], bottom_right[1])
    return None
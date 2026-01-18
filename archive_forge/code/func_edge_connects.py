import inspect
import itertools
import string
import html
from collections.abc import Sequence
from dataclasses import dataclass
from operator import itemgetter
from . import (
def edge_connects(p1, p2) -> bool:

    def edges_to_set(edges):
        return set(map(obj_to_bbox, edges))
    if p1[0] == p2[0]:
        common = edges_to_set(intersections[p1]['v']).intersection(edges_to_set(intersections[p2]['v']))
        if len(common):
            return True
    if p1[1] == p2[1]:
        common = edges_to_set(intersections[p1]['h']).intersection(edges_to_set(intersections[p2]['h']))
        if len(common):
            return True
    return False
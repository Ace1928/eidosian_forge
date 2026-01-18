import inspect
import itertools
import string
import html
from collections.abc import Sequence
from dataclasses import dataclass
from operator import itemgetter
from . import (
def intersections_to_cells(intersections):
    """
    Given a list of points (`intersections`), return all rectangular "cells"
    that those points describe.

    `intersections` should be a dictionary with (x0, top) tuples as keys,
    and a list of edge objects as values. The edge objects should correspond
    to the edges that touch the intersection.
    """

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
    points = list(sorted(intersections.keys()))
    n_points = len(points)

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
    cell_gen = (find_smallest_cell(points, i) for i in range(len(points)))
    return list(filter(None, cell_gen))
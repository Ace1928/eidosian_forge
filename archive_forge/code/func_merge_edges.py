import inspect
import itertools
import string
import html
from collections.abc import Sequence
from dataclasses import dataclass
from operator import itemgetter
from . import (
def merge_edges(edges, snap_x_tolerance, snap_y_tolerance, join_x_tolerance, join_y_tolerance):
    """
    Using the `snap_edges` and `join_edge_group` methods above,
    merge a list of edges into a more "seamless" list.
    """

    def get_group(edge):
        if edge['orientation'] == 'h':
            return ('h', edge['top'])
        else:
            return ('v', edge['x0'])
    if snap_x_tolerance > 0 or snap_y_tolerance > 0:
        edges = snap_edges(edges, snap_x_tolerance, snap_y_tolerance)
    _sorted = sorted(edges, key=get_group)
    edge_groups = itertools.groupby(_sorted, key=get_group)
    edge_gen = (join_edge_group(items, k[0], join_x_tolerance if k[0] == 'h' else join_y_tolerance) for k, items in edge_groups)
    edges = list(itertools.chain(*edge_gen))
    return edges
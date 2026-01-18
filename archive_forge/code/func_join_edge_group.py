import inspect
import itertools
import string
import html
from collections.abc import Sequence
from dataclasses import dataclass
from operator import itemgetter
from . import (
def join_edge_group(edges, orientation: str, tolerance=DEFAULT_JOIN_TOLERANCE):
    """
    Given a list of edges along the same infinite line, join those that
    are within `tolerance` pixels of one another.
    """
    if orientation == 'h':
        min_prop, max_prop = ('x0', 'x1')
    elif orientation == 'v':
        min_prop, max_prop = ('top', 'bottom')
    else:
        raise ValueError("Orientation must be 'v' or 'h'")
    sorted_edges = list(sorted(edges, key=itemgetter(min_prop)))
    joined = [sorted_edges[0]]
    for e in sorted_edges[1:]:
        last = joined[-1]
        if e[min_prop] <= last[max_prop] + tolerance:
            if e[max_prop] > last[max_prop]:
                joined[-1] = resize_object(last, max_prop, e[max_prop])
        else:
            joined.append(e)
    return joined
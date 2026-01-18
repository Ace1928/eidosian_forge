import operator
import threading
import functools
import itertools
import contextlib
import collections
from ..autoray import (
from .draw import (
def ascend(lz):
    """Generate each unique computational node, from leaves to root. I.e. a
    topological ordering of the computational graph. Moreover, the nodes
    are visited 'deepest first'.

    Parameters
    ----------
    lz : pytree of LazyArray
        The output node(s) of the computational graph to ascend to.

    Yields
    ------
    LazyArray
    """
    queue = to_queue(lz)
    seen = set()
    ready = set()
    while queue:
        node = queue[-1]
        need_to_visit = [c for c in node._deps if id(c) not in ready]
        if need_to_visit:
            need_to_visit.sort(key=get_depth)
            queue.extend(need_to_visit)
        else:
            node = queue.pop()
            nid = id(node)
            ready.add(nid)
            if nid not in seen:
                yield node
                seen.add(nid)
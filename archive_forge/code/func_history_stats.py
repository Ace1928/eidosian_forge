import operator
import threading
import functools
import itertools
import contextlib
import collections
from ..autoray import (
from .draw import (
def history_stats(self, fn):
    """Compute aggregate statistics about the computational graph.

        Parameters
        ----------
        fn : callable or str
            Function to apply to each node in the computational graph. If a
            string, one of 'count', 'sizein', 'sizeout' can be used to count
            the number of nodes, the total size of the inputs, or the total
            size of each output respectively.

        Returns
        -------
        stats : dict
            Dictionary mapping function names to the aggregate statistics.
        """
    if not callable(fn):
        if fn == 'count':

            def fn(node):
                return 1
        elif fn == 'sizein':

            def fn(node):
                return sum((child.size for child in node.deps))
        elif fn == 'sizeout':

            def fn(node):
                return node.size
    stats = collections.defaultdict(int)
    for node in self.descend():
        node_cost = fn(node)
        if node_cost is not None:
            stats[node.fn_name] += fn(node)
    return dict(stats)
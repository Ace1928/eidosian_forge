from .graphs import ReducedGraph, Digraph, Poset
from collections import deque
import operator
def spell(self, int_list):
    """
        Convert a sequence of integers to a string.
        """
    if int_list:
        return self.separator.join((self[x] for x in int_list))
    return self[0]
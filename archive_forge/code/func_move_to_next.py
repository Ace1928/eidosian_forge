from collections import deque
import networkx as nx
def move_to_next(self):
    try:
        self._curr = next(self._it)
    except StopIteration:
        self._rewind()
        raise
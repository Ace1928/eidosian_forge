import itertools
from collections import defaultdict, deque
import networkx as nx
from networkx.utils import arbitrary_element, py_random_state
def clear_color(self, adj_entry, color):
    if adj_entry.col_prev is None:
        self.adj_color[color] = adj_entry.col_next
    else:
        adj_entry.col_prev.col_next = adj_entry.col_next
    if adj_entry.col_next is not None:
        adj_entry.col_next.col_prev = adj_entry.col_prev
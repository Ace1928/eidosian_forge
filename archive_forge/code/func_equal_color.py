import itertools
from collections import Counter, defaultdict
from functools import reduce, wraps
def equal_color(node1, node2):
    return node_edge_colors[node1] == node_edge_colors[node2]
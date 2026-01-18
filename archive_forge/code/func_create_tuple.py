import heapq
from collections import deque
from functools import partial
from itertools import chain, combinations, product, starmap
from math import gcd
import networkx as nx
from networkx.utils import arbitrary_element, not_implemented_for, pairwise
def create_tuple(node):
    return (key(node), nodeid_map[node], node)
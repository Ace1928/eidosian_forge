from collections import Counter, defaultdict
from itertools import combinations, product
from math import inf
import networkx as nx
from networkx.utils import not_implemented_for, pairwise
def tailhead(edge):
    if edge[-1] == 'reverse':
        return (edge[1], edge[0])
    return edge[:2]
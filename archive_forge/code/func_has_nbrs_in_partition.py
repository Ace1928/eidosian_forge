from collections import defaultdict
from itertools import combinations
from operator import itemgetter
import networkx as nx
from networkx.algorithms.flow import edmonds_karp
from networkx.utils import not_implemented_for
def has_nbrs_in_partition(G, node, partition):
    return any((n in partition for n in G[node]))
import heapq
import math
from itertools import chain, combinations, zip_longest
from operator import itemgetter
import networkx as nx
from networkx.utils import py_random_state, random_weighted_sample
@nx._dispatch(graphs=None)
def degree_sequence_tree(deg_sequence, create_using=None):
    """Make a tree for the given degree sequence.

    A tree has #nodes-#edges=1 so
    the degree sequence must have
    len(deg_sequence)-sum(deg_sequence)/2=1
    """
    degree_sum = sum(deg_sequence)
    if degree_sum % 2 != 0:
        msg = 'Invalid degree sequence: sum of degrees must be even, not odd'
        raise nx.NetworkXError(msg)
    if len(deg_sequence) - degree_sum // 2 != 1:
        msg = 'Invalid degree sequence: tree must have number of nodes equal to one less than the number of edges'
        raise nx.NetworkXError(msg)
    G = nx.empty_graph(0, create_using)
    if G.is_directed():
        raise nx.NetworkXError('Directed Graph not supported')
    deg = sorted((s for s in deg_sequence if s > 1), reverse=True)
    n = len(deg) + 2
    nx.add_path(G, range(n))
    last = n
    for source in range(1, n - 1):
        nedges = deg.pop() - 2
        for target in range(last, last + nedges):
            G.add_edge(source, target)
        last += nedges
    if len(G) > len(deg_sequence):
        G.remove_node(0)
    return G
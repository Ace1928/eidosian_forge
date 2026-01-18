from collections import defaultdict
from collections.abc import Mapping, Set
from itertools import combinations_with_replacement
import networkx as nx
from networkx.utils import UnionFind, arbitrary_element, not_implemented_for
def generate_lca_from_pairs(G, pairs):
    ancestor_cache = {}
    for v, w in pairs:
        if v not in ancestor_cache:
            ancestor_cache[v] = nx.ancestors(G, v)
            ancestor_cache[v].add(v)
        if w not in ancestor_cache:
            ancestor_cache[w] = nx.ancestors(G, w)
            ancestor_cache[w].add(w)
        common_ancestors = ancestor_cache[v] & ancestor_cache[w]
        if common_ancestors:
            common_ancestor = next(iter(common_ancestors))
            while True:
                successor = None
                for lower_ancestor in G.successors(common_ancestor):
                    if lower_ancestor in common_ancestors:
                        successor = lower_ancestor
                        break
                if successor is None:
                    break
                common_ancestor = successor
            yield ((v, w), common_ancestor)
from itertools import chain, islice, repeat
from math import ceil, sqrt
import networkx as nx
from networkx.utils import not_implemented_for
def make_root(self, q):
    """
        Make a node q the root of its containing subtree.
        """
    ancestors = []
    while q is not None:
        ancestors.append(q)
        q = self.parent[q]
    ancestors.reverse()
    for p, q in zip(ancestors, islice(ancestors, 1, None)):
        size_p = self.subtree_size[p]
        last_p = self.last_descendent_dft[p]
        prev_q = self.prev_node_dft[q]
        last_q = self.last_descendent_dft[q]
        next_last_q = self.next_node_dft[last_q]
        self.parent[p] = q
        self.parent[q] = None
        self.parent_edge[p] = self.parent_edge[q]
        self.parent_edge[q] = None
        self.subtree_size[p] = size_p - self.subtree_size[q]
        self.subtree_size[q] = size_p
        self.next_node_dft[prev_q] = next_last_q
        self.prev_node_dft[next_last_q] = prev_q
        self.next_node_dft[last_q] = q
        self.prev_node_dft[q] = last_q
        if last_p == last_q:
            self.last_descendent_dft[p] = prev_q
            last_p = prev_q
        self.prev_node_dft[p] = last_q
        self.next_node_dft[last_q] = p
        self.next_node_dft[last_p] = q
        self.prev_node_dft[q] = last_p
        self.last_descendent_dft[q] = last_p
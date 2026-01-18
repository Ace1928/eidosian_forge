from itertools import chain, islice, repeat
from math import ceil, sqrt
import networkx as nx
from networkx.utils import not_implemented_for
def find_apex(self, p, q):
    """
        Find the lowest common ancestor of nodes p and q in the spanning tree.
        """
    size_p = self.subtree_size[p]
    size_q = self.subtree_size[q]
    while True:
        while size_p < size_q:
            p = self.parent[p]
            size_p = self.subtree_size[p]
        while size_p > size_q:
            q = self.parent[q]
            size_q = self.subtree_size[q]
        if size_p == size_q:
            if p != q:
                p = self.parent[p]
                size_p = self.subtree_size[p]
                q = self.parent[q]
                size_q = self.subtree_size[q]
            else:
                return p
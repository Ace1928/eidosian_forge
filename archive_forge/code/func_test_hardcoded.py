import random
import time
import networkx as nx
from networkx.algorithms.isomorphism.tree_isomorphism import (
from networkx.classes.function import is_directed
def test_hardcoded():
    print('hardcoded test')
    edges_1 = [('a', 'b'), ('a', 'c'), ('a', 'd'), ('b', 'e'), ('b', 'f'), ('e', 'j'), ('e', 'k'), ('c', 'g'), ('c', 'h'), ('g', 'm'), ('d', 'i'), ('f', 'l')]
    edges_2 = [('v', 'y'), ('v', 'z'), ('u', 'x'), ('q', 'u'), ('q', 'v'), ('p', 't'), ('n', 'p'), ('n', 'q'), ('n', 'o'), ('o', 'r'), ('o', 's'), ('s', 'w')]
    isomorphism1 = [('a', 'n'), ('b', 'q'), ('c', 'o'), ('d', 'p'), ('e', 'v'), ('f', 'u'), ('g', 's'), ('h', 'r'), ('i', 't'), ('j', 'y'), ('k', 'z'), ('l', 'x'), ('m', 'w')]
    isomorphism2 = [('a', 'n'), ('b', 'q'), ('c', 'o'), ('d', 'p'), ('e', 'v'), ('f', 'u'), ('g', 's'), ('h', 'r'), ('i', 't'), ('j', 'z'), ('k', 'y'), ('l', 'x'), ('m', 'w')]
    t1 = nx.Graph()
    t1.add_edges_from(edges_1)
    root1 = 'a'
    t2 = nx.Graph()
    t2.add_edges_from(edges_2)
    root2 = 'n'
    isomorphism = sorted(rooted_tree_isomorphism(t1, root1, t2, root2))
    assert isomorphism in (isomorphism1, isomorphism2)
    assert check_isomorphism(t1, t2, isomorphism)
    t1 = nx.DiGraph()
    t1.add_edges_from(edges_1)
    root1 = 'a'
    t2 = nx.DiGraph()
    t2.add_edges_from(edges_2)
    root2 = 'n'
    isomorphism = sorted(rooted_tree_isomorphism(t1, root1, t2, root2))
    assert isomorphism in (isomorphism1, isomorphism2)
    assert check_isomorphism(t1, t2, isomorphism)
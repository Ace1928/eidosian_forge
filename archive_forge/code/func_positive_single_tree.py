import random
import time
import networkx as nx
from networkx.algorithms.isomorphism.tree_isomorphism import (
from networkx.classes.function import is_directed
def positive_single_tree(t1):
    assert nx.is_tree(t1)
    nodes1 = list(t1.nodes())
    nodes2 = nodes1.copy()
    random.shuffle(nodes2)
    someisomorphism = list(zip(nodes1, nodes2))
    map1to2 = dict(someisomorphism)
    edges2 = [random_swap((map1to2[u], map1to2[v])) for u, v in t1.edges()]
    random.shuffle(edges2)
    t2 = nx.Graph()
    t2.add_edges_from(edges2)
    isomorphism = tree_isomorphism(t1, t2)
    assert len(isomorphism) > 0
    assert check_isomorphism(t1, t2, isomorphism)
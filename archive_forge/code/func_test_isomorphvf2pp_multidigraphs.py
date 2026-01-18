import importlib.resources
import os
import random
import struct
import networkx as nx
from networkx.algorithms import isomorphism as iso
def test_isomorphvf2pp_multidigraphs():
    g = nx.MultiDiGraph({0: [1, 1, 2, 2, 3], 1: [2, 3, 3], 2: [3]})
    h = nx.MultiDiGraph({0: [1, 1, 2, 2, 3], 1: [2, 3, 3], 3: [2]})
    assert not nx.vf2pp_is_isomorphic(g, h)
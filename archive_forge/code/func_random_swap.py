import random
import time
import networkx as nx
from networkx.algorithms.isomorphism.tree_isomorphism import (
from networkx.classes.function import is_directed
def random_swap(t):
    a, b = t
    if random.randint(0, 1) == 1:
        return (a, b)
    else:
        return (b, a)
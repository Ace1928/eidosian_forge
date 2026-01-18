import numbers
from collections import Counter
import networkx as nx
from networkx.generators.classic import empty_graph
from networkx.utils import discrete_sequence, py_random_state, weighted_choice
def kernel(x):
    return x
import networkx as nx
from networkx.exception import NetworkXError
from networkx.utils import not_implemented_for
def k_filter(v, k, c):
    return c[v] == k
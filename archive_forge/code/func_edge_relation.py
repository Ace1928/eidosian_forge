from itertools import chain, combinations, permutations, product
import networkx as nx
from networkx import density
from networkx.exception import NetworkXException
from networkx.utils import arbitrary_element
def edge_relation(b, c):
    return any((v in G[u] for u, v in product(b, c)))
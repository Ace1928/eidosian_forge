import pytest
import networkx as nx
from networkx.utils import arbitrary_element, edges_equal, nodes_equal
def same_parity(b, c):
    return arbitrary_element(b) % 2 == arbitrary_element(c) % 2
import random
import pytest
import networkx as nx
from networkx.utils import arbitrary_element, graphs_equal
def source_label(v):
    return T.nodes[v]['source']
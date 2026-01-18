import itertools
import numbers
import networkx as nx
from networkx.classes import Graph
from networkx.exception import NetworkXError
from networkx.utils import nodes_or_number, pairwise
@nx._dispatch(graphs=None)
def null_graph(create_using=None):
    """Returns the Null graph with no nodes or edges.

    See empty_graph for the use of create_using.

    """
    G = empty_graph(0, create_using)
    return G
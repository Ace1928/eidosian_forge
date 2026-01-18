from collections import Counter
from itertools import chain
import networkx as nx
from networkx.utils import not_implemented_for, pairwise
def is_frozen(G):
    """Returns True if graph is frozen.

    Parameters
    ----------
    G : graph
      A NetworkX graph

    See Also
    --------
    freeze
    """
    try:
        return G.frozen
    except AttributeError:
        return False
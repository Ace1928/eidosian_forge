from the graph class, we short-cut the chain by returning a
import networkx as nx
from networkx.classes.coreviews import (
from networkx.classes.filters import no_filter
from networkx.exception import NetworkXError
from networkx.utils import deprecate_positional_args, not_implemented_for
def reverse_edge(u, v, k=None):
    return filter_edge(v, u)
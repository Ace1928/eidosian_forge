from copy import deepcopy
from functools import cached_property
import networkx as nx
from networkx import convert
from networkx.classes.coreviews import AdjacencyView
from networkx.classes.graph import Graph
from networkx.classes.reportviews import (
from networkx.exception import NetworkXError
def has_predecessor(self, u, v):
    """Returns True if node u has predecessor v.

        This is true if graph has the edge u<-v.
        """
    return u in self._pred and v in self._pred[u]
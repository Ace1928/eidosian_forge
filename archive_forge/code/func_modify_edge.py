import pickle
from copy import deepcopy
import pytest
import networkx as nx
from networkx.classes import reportviews as rv
from networkx.classes.reportviews import NodeDataView
def modify_edge(self, G, e, **kwds):
    if len(e) == 2:
        e = e + (0,)
    G._adj[e[0]][e[1]][e[2]].update(kwds)
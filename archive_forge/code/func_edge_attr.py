import pytest
import networkx as nx
def edge_attr(u, v):
    return G[u][v].get('thickness', 0.5)
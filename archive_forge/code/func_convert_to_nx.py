import pickle
import pytest
import networkx as nx
@staticmethod
def convert_to_nx(obj, *, name=None):
    if type(obj) is nx.Graph:
        return obj
    return nx.Graph(obj)
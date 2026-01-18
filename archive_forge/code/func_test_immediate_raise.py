import os
import pathlib
import random
import tempfile
import pytest
import networkx as nx
from networkx.utils.decorators import (
from networkx.utils.misc import PythonRandomInterface
def test_immediate_raise(self):

    @not_implemented_for('directed')
    def yield_nodes(G):
        yield from G
    G = nx.Graph([(1, 2)])
    D = nx.DiGraph()
    with pytest.raises(nx.NetworkXNotImplemented):
        node_iter = yield_nodes(D)
    with pytest.raises(nx.NetworkXNotImplemented):
        node_iter = yield_nodes(D)
    node_iter = yield_nodes(G)
    next(node_iter)
    next(node_iter)
    with pytest.raises(StopIteration):
        next(node_iter)
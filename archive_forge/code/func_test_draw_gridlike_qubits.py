import itertools
from unittest.mock import MagicMock
import cirq
import networkx as nx
import pytest
from cirq import (
@pytest.mark.parametrize('tilted', [True, False])
def test_draw_gridlike_qubits(tilted):
    graph = nx.grid_2d_graph(3, 3)
    graph = nx.relabel_nodes(graph, {(r, c): cirq.GridQubit(r, c) for r, c in sorted(graph.nodes)})
    ax = MagicMock()
    pos = draw_gridlike(graph, tilted=tilted, ax=ax)
    ax.scatter.assert_called()
    for q, _ in pos.items():
        assert 0 <= q.row < 3
        assert 0 <= q.col < 3
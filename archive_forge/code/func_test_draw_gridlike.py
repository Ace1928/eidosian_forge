import itertools
from unittest.mock import MagicMock
import cirq
import networkx as nx
import pytest
from cirq import (
@pytest.mark.parametrize('tilted', [True, False])
def test_draw_gridlike(tilted):
    graph = nx.grid_2d_graph(3, 3)
    ax = MagicMock()
    pos = draw_gridlike(graph, tilted=tilted, ax=ax)
    ax.scatter.assert_called()
    for (row, column), _ in pos.items():
        assert 0 <= row < 3
        assert 0 <= column < 3
import itertools
from unittest.mock import MagicMock
import cirq
import networkx as nx
import pytest
from cirq import (
def test_line_topology():
    n = 10
    topo = LineTopology(n)
    assert topo.n_nodes == n
    assert topo.n_nodes == topo.graph.number_of_nodes()
    assert all((1 <= topo.graph.degree[node] <= 2 for node in topo.graph.nodes))
    assert topo.name == 'line-10'
    ax = MagicMock()
    topo.draw(ax=ax)
    ax.scatter.assert_called()
    with pytest.raises(ValueError, match='greater than 1.*'):
        _ = LineTopology(1)
    assert LineTopology(2).n_nodes == 2
    assert LineTopology(2).graph.number_of_nodes() == 2
    mapping = topo.nodes_to_linequbits(offset=3)
    assert sorted(mapping.keys()) == list(range(n))
    assert all((isinstance(q, cirq.LineQubit) for q in mapping.values()))
    assert all((mapping[x] == cirq.LineQubit(x + 3) for x in mapping))
    cirq.testing.assert_equivalent_repr(topo)
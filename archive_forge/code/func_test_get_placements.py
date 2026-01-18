import itertools
from unittest.mock import MagicMock
import cirq
import networkx as nx
import pytest
from cirq import (
def test_get_placements():
    topo = TiltedSquareLattice(4, 2)
    syc23 = TiltedSquareLattice(8, 4).graph
    placements = get_placements(syc23, topo.graph)
    assert len(placements) == 12
    axes = [MagicMock() for _ in range(4)]
    draw_placements(syc23, topo.graph, placements[::3], axes=axes)
    for ax in axes:
        ax.scatter.assert_called()
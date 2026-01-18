from itertools import combinations
import pytest
from networkx import DiGraph
from networkx.algorithms.tournament import (
def test_hamiltonian_empty_graph():
    path = hamiltonian_path(DiGraph())
    assert len(path) == 0
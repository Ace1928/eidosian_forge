import itertools
from unittest.mock import MagicMock
import cirq
import networkx as nx
import pytest
from cirq import (
def test_bad_tilted_square_lattice():
    with pytest.raises(ValueError):
        _ = TiltedSquareLattice(0, 3)
    with pytest.raises(ValueError):
        _ = TiltedSquareLattice(3, 0)
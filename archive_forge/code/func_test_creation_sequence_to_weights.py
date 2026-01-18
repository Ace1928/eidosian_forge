import pytest
import networkx as nx
import networkx.algorithms.threshold as nxt
from networkx.algorithms.isomorphism.isomorph import graph_could_be_isomorphic
def test_creation_sequence_to_weights(self):
    assert nxt.creation_sequence_to_weights([3, 1, 2]) == [0.5, 0.5, 0.5, 0.25, 0.75, 0.75]
    assert pytest.raises(TypeError, nxt.creation_sequence_to_weights, [3.0, 1.0, 2.0])
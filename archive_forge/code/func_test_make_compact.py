import pytest
import networkx as nx
import networkx.algorithms.threshold as nxt
from networkx.algorithms.isomorphism.isomorph import graph_could_be_isomorphic
def test_make_compact(self):
    assert nxt.make_compact(['d', 'd', 'd', 'i', 'd', 'd']) == [3, 1, 2]
    assert nxt.make_compact([3, 1, 2]) == [3, 1, 2]
    assert pytest.raises(TypeError, nxt.make_compact, [3.0, 1.0, 2.0])
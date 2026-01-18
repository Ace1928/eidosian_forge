import pytest
from networkx.exception import NetworkXError
from networkx.generators.duplication import (
def test_minimum_desired_nodes(self):
    with pytest.raises(NetworkXError, match='.*n must be greater than or equal to 2'):
        duplication_divergence_graph(1, p=1)
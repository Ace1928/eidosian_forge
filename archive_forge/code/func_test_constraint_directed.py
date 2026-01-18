import math
import pytest
import networkx as nx
from networkx.classes.tests import dispatch_interface
@pytest.mark.parametrize('wrapper', [lambda x: x, dispatch_interface.convert])
def test_constraint_directed(self, wrapper):
    constraint = nx.constraint(wrapper(self.D))
    assert constraint[0] == pytest.approx(1.003, abs=0.001)
    assert constraint[1] == pytest.approx(1.003, abs=0.001)
    assert constraint[2] == pytest.approx(1.389, abs=0.001)
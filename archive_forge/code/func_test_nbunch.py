import pickle
from copy import deepcopy
import pytest
import networkx as nx
from networkx.classes import reportviews as rv
from networkx.classes.reportviews import NodeDataView
def test_nbunch(self):
    dv = self.dview(self.G)
    dvn = dv(0)
    assert dvn == 0
    dvn = dv([2, 3])
    assert sorted(dvn) == [(2, 1), (3, 3)]
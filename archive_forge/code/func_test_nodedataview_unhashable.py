import pickle
from copy import deepcopy
import pytest
import networkx as nx
from networkx.classes import reportviews as rv
from networkx.classes.reportviews import NodeDataView
def test_nodedataview_unhashable():
    G = nx.path_graph(9)
    G.nodes[3]['foo'] = 'bar'
    nvs = [G.nodes.data()]
    nvs.append(G.nodes.data(True))
    H = G.copy()
    H.nodes[4]['foo'] = {1, 2, 3}
    nvs.append(H.nodes.data(True))
    for nv in nvs:
        pytest.raises(TypeError, set, nv)
        pytest.raises(TypeError, eval, 'nv | nv', locals())
    Gn = G.nodes.data(False)
    set(Gn)
    Gn | Gn
    Gn = G.nodes.data('foo')
    set(Gn)
    Gn | Gn
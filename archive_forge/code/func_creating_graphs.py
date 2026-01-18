from ..nbs import NetworkBasedStatistic
from ....utils.misc import package_check
import numpy as np
import networkx as nx
import pickle
import pytest
@pytest.fixture()
def creating_graphs(tmpdir):
    graphlist = []
    graphnames = ['name' + str(i) for i in range(6)]
    for idx, name in enumerate(graphnames):
        graph = np.random.rand(10, 10)
        G = nx.from_numpy_array(graph)
        out_file = tmpdir.strpath + graphnames[idx] + '.pck'
        with open(out_file, 'wb') as f:
            pickle.dump(G, f, pickle.HIGHEST_PROTOCOL)
        graphlist.append(out_file)
    return graphlist
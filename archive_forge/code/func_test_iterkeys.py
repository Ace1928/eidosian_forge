import pickle
from copy import deepcopy
import pytest
import networkx as nx
from networkx.classes import reportviews as rv
from networkx.classes.reportviews import NodeDataView
def test_iterkeys(self):
    G = self.G
    evr = self.eview(G)
    ev = evr(keys=True)
    for u, v, k in ev:
        pass
    assert k == 0
    ev = evr(keys=True, data='foo', default=1)
    for u, v, k, wt in ev:
        pass
    assert wt == 1
    self.modify_edge(G, (2, 3, 0), foo='bar')
    ev = evr(keys=True, data=True)
    for e in ev:
        assert len(e) == 4
        print('edge:', e)
        if set(e[:2]) == {2, 3}:
            print(self.G._adj[2][3])
            assert e[2] == 0
            assert e[3] == {'foo': 'bar'}
            checked = True
        elif set(e[:3]) == {1, 2, 3}:
            assert e[2] == 3
            assert e[3] == {'foo': 'bar'}
            checked_multi = True
        else:
            assert e[2] == 0
            assert e[3] == {}
    assert checked
    assert checked_multi
    ev = evr(keys=True, data='foo', default=1)
    for e in ev:
        if set(e[:2]) == {1, 2} and e[2] == 3:
            assert e[3] == 'bar'
        if set(e[:2]) == {1, 2} and e[2] == 0:
            assert e[3] == 1
        if set(e[:2]) == {2, 3}:
            assert e[2] == 0
            assert e[3] == 'bar'
            assert len(e) == 4
            checked_wt = True
    assert checked_wt
    ev = evr(keys=True)
    for e in ev:
        assert len(e) == 3
    elist = sorted([(i, i + 1, 0) for i in range(8)] + [(1, 2, 3)])
    assert sorted(ev) == elist
    ev = evr((1, 2), 'foo', keys=True, default=1)
    with pytest.raises(TypeError):
        evr((1, 2), 'foo', True, 1)
    with pytest.raises(TypeError):
        evr((1, 2), 'foo', True, default=1)
    for e in ev:
        if set(e[:2]) == {1, 2}:
            assert e[2] in {0, 3}
            if e[2] == 3:
                assert e[3] == 'bar'
            else:
                assert e[3] == 1
    if G.is_directed():
        assert len(list(ev)) == 3
    else:
        assert len(list(ev)) == 4
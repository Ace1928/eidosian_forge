import pytest
import networkx as nx
from networkx.utils import pairwise
def test_dijkstra(self):
    D, P = nx.single_source_dijkstra(self.XG, 's')
    validate_path(self.XG, 's', 'v', 9, P['v'])
    assert D['v'] == 9
    validate_path(self.XG, 's', 'v', 9, nx.single_source_dijkstra_path(self.XG, 's')['v'])
    assert dict(nx.single_source_dijkstra_path_length(self.XG, 's'))['v'] == 9
    validate_path(self.XG, 's', 'v', 9, nx.single_source_dijkstra(self.XG, 's')[1]['v'])
    validate_path(self.MXG, 's', 'v', 9, nx.single_source_dijkstra_path(self.MXG, 's')['v'])
    GG = self.XG.to_undirected()
    GG['u']['x']['weight'] = 2
    D, P = nx.single_source_dijkstra(GG, 's')
    validate_path(GG, 's', 'v', 8, P['v'])
    assert D['v'] == 8
    validate_path(GG, 's', 'v', 8, nx.dijkstra_path(GG, 's', 'v'))
    assert nx.dijkstra_path_length(GG, 's', 'v') == 8
    validate_path(self.XG2, 1, 3, 4, nx.dijkstra_path(self.XG2, 1, 3))
    validate_path(self.XG3, 0, 3, 15, nx.dijkstra_path(self.XG3, 0, 3))
    assert nx.dijkstra_path_length(self.XG3, 0, 3) == 15
    validate_path(self.XG4, 0, 2, 4, nx.dijkstra_path(self.XG4, 0, 2))
    assert nx.dijkstra_path_length(self.XG4, 0, 2) == 4
    validate_path(self.MXG4, 0, 2, 4, nx.dijkstra_path(self.MXG4, 0, 2))
    validate_path(self.G, 's', 'v', 2, nx.single_source_dijkstra(self.G, 's', 'v')[1])
    validate_path(self.G, 's', 'v', 2, nx.single_source_dijkstra(self.G, 's')[1]['v'])
    validate_path(self.G, 's', 'v', 2, nx.dijkstra_path(self.G, 's', 'v'))
    assert nx.dijkstra_path_length(self.G, 's', 'v') == 2
    pytest.raises(nx.NetworkXNoPath, nx.dijkstra_path, self.G, 's', 'moon')
    pytest.raises(nx.NetworkXNoPath, nx.dijkstra_path_length, self.G, 's', 'moon')
    validate_path(self.cycle, 0, 3, 3, nx.dijkstra_path(self.cycle, 0, 3))
    validate_path(self.cycle, 0, 4, 3, nx.dijkstra_path(self.cycle, 0, 4))
    assert nx.single_source_dijkstra(self.cycle, 0, 0) == (0, [0])
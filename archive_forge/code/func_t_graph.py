import networkx as nx
from networkx.utils import py_random_state
def t_graph(self):
    """Generates the core mesh network of tier one nodes of a AS graph.

        Returns
        -------
        G: Networkx Graph
            Core network
        """
    self.G = nx.Graph()
    for i in range(self.n_t):
        self.G.add_node(i, type='T')
        for r in self.regions:
            self.regions[r].add(i)
        for j in self.G.nodes():
            if i != j:
                self.add_edge(i, j, 'peer')
        self.customers[i] = set()
        self.providers[i] = set()
    return self.G
import networkx as nx
from networkx.utils import py_random_state
def add_m_peering_link(self, m, to_kind):
    """Add a peering link between two middle tier (M) nodes.

        Target node j is drawn considering a preferential attachment based on
        other M node peering degree.

        Parameters
        ----------
        m: object
            Node identifier
        to_kind: string
            type for target node j (must be always M)

        Returns
        -------
        success: boolean
        """
    node_options = self.nodes['M'].difference(self.customers[m])
    node_options = node_options.difference(self.providers[m])
    if m in node_options:
        node_options.remove(m)
    for j in self.G.neighbors(m):
        if j in node_options:
            node_options.remove(j)
    if len(node_options) > 0:
        j = self.choose_peer_pref_attach(node_options)
        self.add_edge(m, j, 'peer')
        self.G.nodes[m]['peers'] += 1
        self.G.nodes[j]['peers'] += 1
        return True
    else:
        return False
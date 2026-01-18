import itertools as it
from functools import partial
import networkx as nx
from networkx.utils import arbitrary_element, not_implemented_for
def k_edge_components(self, k):
    """Queries the auxiliary graph for k-edge-connected components.

        Parameters
        ----------
        k : Integer
            Desired edge connectivity

        Returns
        -------
        k_edge_components : a generator of k-edge-ccs

        Notes
        -----
        Given the auxiliary graph, the k-edge-connected components can be
        determined in linear time by removing all edges with weights less than
        k from the auxiliary graph.  The resulting connected components are the
        k-edge-ccs in the original graph.
        """
    if k < 1:
        raise ValueError('k cannot be less than 1')
    A = self.A
    aux_weights = nx.get_edge_attributes(A, 'weight')
    R = nx.Graph()
    R.add_nodes_from(A.nodes())
    R.add_edges_from((e for e, w in aux_weights.items() if w >= k))
    yield from nx.connected_components(R)
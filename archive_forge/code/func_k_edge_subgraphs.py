import itertools as it
from functools import partial
import networkx as nx
from networkx.utils import arbitrary_element, not_implemented_for
def k_edge_subgraphs(self, k):
    """Queries the auxiliary graph for k-edge-connected subgraphs.

        Parameters
        ----------
        k : Integer
            Desired edge connectivity

        Returns
        -------
        k_edge_subgraphs : a generator of k-edge-subgraphs

        Notes
        -----
        Refines the k-edge-ccs into k-edge-subgraphs. The running time is more
        than $O(|V|)$.

        For single values of k it is faster to use `nx.k_edge_subgraphs`.
        But for multiple values of k, it can be faster to build AuxGraph and
        then use this method.
        """
    if k < 1:
        raise ValueError('k cannot be less than 1')
    H = self.H
    A = self.A
    aux_weights = nx.get_edge_attributes(A, 'weight')
    R = nx.Graph()
    R.add_nodes_from(A.nodes())
    R.add_edges_from((e for e, w in aux_weights.items() if w >= k))
    for cc in nx.connected_components(R):
        if len(cc) < k:
            for node in cc:
                yield {node}
        else:
            C = H.subgraph(cc)
            yield from k_edge_subgraphs(C, k)
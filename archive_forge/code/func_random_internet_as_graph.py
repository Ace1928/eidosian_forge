import networkx as nx
from networkx.utils import py_random_state
@py_random_state(1)
@nx._dispatch(graphs=None)
def random_internet_as_graph(n, seed=None):
    """Generates a random undirected graph resembling the Internet AS network

    Parameters
    ----------
    n: integer in [1000, 10000]
        Number of graph nodes
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Returns
    -------
    G: Networkx Graph object
        A randomly generated undirected graph

    Notes
    -----
    This algorithm returns an undirected graph resembling the Internet
    Autonomous System (AS) network, it uses the approach by Elmokashfi et al.
    [1]_ and it grants the properties described in the related paper [1]_.

    Each node models an autonomous system, with an attribute 'type' specifying
    its kind; tier-1 (T), mid-level (M), customer (C) or content-provider (CP).
    Each edge models an ADV communication link (hence, bidirectional) with
    attributes:

      - type: transit|peer, the kind of commercial agreement between nodes;
      - customer: <node id>, the identifier of the node acting as customer
        ('none' if type is peer).

    References
    ----------
    .. [1] A. Elmokashfi, A. Kvalbein and C. Dovrolis, "On the Scalability of
       BGP: The Role of Topology Growth," in IEEE Journal on Selected Areas
       in Communications, vol. 28, no. 8, pp. 1250-1261, October 2010.
    """
    GG = AS_graph_generator(n, seed)
    G = GG.generate()
    return G
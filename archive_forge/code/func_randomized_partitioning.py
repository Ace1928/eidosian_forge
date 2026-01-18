import networkx as nx
from networkx.utils.decorators import not_implemented_for, py_random_state
@not_implemented_for('directed', 'multigraph')
@py_random_state(1)
@nx._dispatch(edge_attrs='weight')
def randomized_partitioning(G, seed=None, p=0.5, weight=None):
    """Compute a random partitioning of the graph nodes and its cut value.

    A partitioning is calculated by observing each node
    and deciding to add it to the partition with probability `p`,
    returning a random cut and its corresponding value (the
    sum of weights of edges connecting different partitions).

    Parameters
    ----------
    G : NetworkX graph

    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    p : scalar
        Probability for each node to be part of the first partition.
        Should be in [0,1]

    weight : object
        Edge attribute key to use as weight. If not specified, edges
        have weight one.

    Returns
    -------
    cut_size : scalar
        Value of the minimum cut.

    partition : pair of node sets
        A partitioning of the nodes that defines a minimum cut.
    """
    cut = {node for node in G.nodes() if seed.random() < p}
    cut_size = nx.algorithms.cut_size(G, cut, weight=weight)
    partition = (cut, G.nodes - cut)
    return (cut_size, partition)
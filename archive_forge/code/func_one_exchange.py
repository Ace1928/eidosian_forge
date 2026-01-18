import networkx as nx
from networkx.utils.decorators import not_implemented_for, py_random_state
@not_implemented_for('directed', 'multigraph')
@py_random_state(2)
@nx._dispatch(edge_attrs='weight')
def one_exchange(G, initial_cut=None, seed=None, weight=None):
    """Compute a partitioning of the graphs nodes and the corresponding cut value.

    Use a greedy one exchange strategy to find a locally maximal cut
    and its value, it works by finding the best node (one that gives
    the highest gain to the cut value) to add to the current cut
    and repeats this process until no improvement can be made.

    Parameters
    ----------
    G : networkx Graph
        Graph to find a maximum cut for.

    initial_cut : set
        Cut to use as a starting point. If not supplied the algorithm
        starts with an empty cut.

    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    weight : object
        Edge attribute key to use as weight. If not specified, edges
        have weight one.

    Returns
    -------
    cut_value : scalar
        Value of the maximum cut.

    partition : pair of node sets
        A partitioning of the nodes that defines a maximum cut.
    """
    if initial_cut is None:
        initial_cut = set()
    cut = set(initial_cut)
    current_cut_size = nx.algorithms.cut_size(G, cut, weight=weight)
    while True:
        nodes = list(G.nodes())
        seed.shuffle(nodes)
        best_node_to_swap = max(nodes, key=lambda v: nx.algorithms.cut_size(G, _swap_node_partition(cut, v), weight=weight), default=None)
        potential_cut = _swap_node_partition(cut, best_node_to_swap)
        potential_cut_size = nx.algorithms.cut_size(G, potential_cut, weight=weight)
        if potential_cut_size > current_cut_size:
            cut = potential_cut
            current_cut_size = potential_cut_size
        else:
            break
    partition = (cut, G.nodes - cut)
    return (current_cut_size, partition)
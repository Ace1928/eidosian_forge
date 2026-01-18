import math
import numbers
from functools import reduce
import networkx as nx
from networkx.utils import nodes_or_number, py_random_state
@py_random_state(3)
@nx._dispatch(name='bipartite_configuration_model', graphs=None)
def configuration_model(aseq, bseq, create_using=None, seed=None):
    """Returns a random bipartite graph from two given degree sequences.

    Parameters
    ----------
    aseq : list
       Degree sequence for node set A.
    bseq : list
       Degree sequence for node set B.
    create_using : NetworkX graph instance, optional
       Return graph of this type.
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    The graph is composed of two partitions. Set A has nodes 0 to
    (len(aseq) - 1) and set B has nodes len(aseq) to (len(bseq) - 1).
    Nodes from set A are connected to nodes in set B by choosing
    randomly from the possible free stubs, one in A and one in B.

    Notes
    -----
    The sum of the two sequences must be equal: sum(aseq)=sum(bseq)
    If no graph type is specified use MultiGraph with parallel edges.
    If you want a graph with no parallel edges use create_using=Graph()
    but then the resulting degree sequences might not be exact.

    The nodes are assigned the attribute 'bipartite' with the value 0 or 1
    to indicate which bipartite set the node belongs to.

    This function is not imported in the main namespace.
    To use it use nx.bipartite.configuration_model
    """
    G = nx.empty_graph(0, create_using, default=nx.MultiGraph)
    if G.is_directed():
        raise nx.NetworkXError('Directed Graph not supported')
    lena = len(aseq)
    lenb = len(bseq)
    suma = sum(aseq)
    sumb = sum(bseq)
    if not suma == sumb:
        raise nx.NetworkXError(f'invalid degree sequences, sum(aseq)!=sum(bseq),{suma},{sumb}')
    G = _add_nodes_with_bipartite_label(G, lena, lenb)
    if len(aseq) == 0 or max(aseq) == 0:
        return G
    stubs = [[v] * aseq[v] for v in range(lena)]
    astubs = [x for subseq in stubs for x in subseq]
    stubs = [[v] * bseq[v - lena] for v in range(lena, lena + lenb)]
    bstubs = [x for subseq in stubs for x in subseq]
    seed.shuffle(astubs)
    seed.shuffle(bstubs)
    G.add_edges_from(([astubs[i], bstubs[i]] for i in range(suma)))
    G.name = 'bipartite_configuration_model'
    return G
import heapq
import networkx as nx
@nx._dispatch(graphs=None)
def is_graphical(sequence, method='eg'):
    """Returns True if sequence is a valid degree sequence.

    A degree sequence is valid if some graph can realize it.

    Parameters
    ----------
    sequence : list or iterable container
        A sequence of integer node degrees

    method : "eg" | "hh"  (default: 'eg')
        The method used to validate the degree sequence.
        "eg" corresponds to the Erdős-Gallai algorithm
        [EG1960]_, [choudum1986]_, and
        "hh" to the Havel-Hakimi algorithm
        [havel1955]_, [hakimi1962]_, [CL1996]_.

    Returns
    -------
    valid : bool
        True if the sequence is a valid degree sequence and False if not.

    Examples
    --------
    >>> G = nx.path_graph(4)
    >>> sequence = (d for n, d in G.degree())
    >>> nx.is_graphical(sequence)
    True

    To test a non-graphical sequence:
    >>> sequence_list = [d for n, d in G.degree()]
    >>> sequence_list[-1] += 1
    >>> nx.is_graphical(sequence_list)
    False

    References
    ----------
    .. [EG1960] Erdős and Gallai, Mat. Lapok 11 264, 1960.
    .. [choudum1986] S.A. Choudum. "A simple proof of the Erdős-Gallai theorem on
       graph sequences." Bulletin of the Australian Mathematical Society, 33,
       pp 67-70, 1986. https://doi.org/10.1017/S0004972700002872
    .. [havel1955] Havel, V. "A Remark on the Existence of Finite Graphs"
       Casopis Pest. Mat. 80, 477-480, 1955.
    .. [hakimi1962] Hakimi, S. "On the Realizability of a Set of Integers as
       Degrees of the Vertices of a Graph." SIAM J. Appl. Math. 10, 496-506, 1962.
    .. [CL1996] G. Chartrand and L. Lesniak, "Graphs and Digraphs",
       Chapman and Hall/CRC, 1996.
    """
    if method == 'eg':
        valid = is_valid_degree_sequence_erdos_gallai(list(sequence))
    elif method == 'hh':
        valid = is_valid_degree_sequence_havel_hakimi(list(sequence))
    else:
        msg = "`method` must be 'eg' or 'hh'"
        raise nx.NetworkXException(msg)
    return valid
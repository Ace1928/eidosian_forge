import heapq
import networkx as nx
@nx._dispatch(graphs=None)
def is_valid_degree_sequence_erdos_gallai(deg_sequence):
    """Returns True if deg_sequence can be realized by a simple graph.

    The validation is done using the Erdős-Gallai theorem [EG1960]_.

    Parameters
    ----------
    deg_sequence : list
        A list of integers

    Returns
    -------
    valid : bool
        True if deg_sequence is graphical and False if not.

    Examples
    --------
    >>> G = nx.Graph([(1, 2), (1, 3), (2, 3), (3, 4), (4, 2), (5, 1), (5, 4)])
    >>> sequence = (d for _, d in G.degree())
    >>> nx.is_valid_degree_sequence_erdos_gallai(sequence)
    True

    To test a non-valid sequence:
    >>> sequence_list = [d for _, d in G.degree()]
    >>> sequence_list[-1] += 1
    >>> nx.is_valid_degree_sequence_erdos_gallai(sequence_list)
    False

    Notes
    -----

    This implementation uses an equivalent form of the Erdős-Gallai criterion.
    Worst-case run time is $O(n)$ where $n$ is the length of the sequence.

    Specifically, a sequence d is graphical if and only if the
    sum of the sequence is even and for all strong indices k in the sequence,

     .. math::

       \\sum_{i=1}^{k} d_i \\leq k(k-1) + \\sum_{j=k+1}^{n} \\min(d_i,k)
             = k(n-1) - ( k \\sum_{j=0}^{k-1} n_j - \\sum_{j=0}^{k-1} j n_j )

    A strong index k is any index where d_k >= k and the value n_j is the
    number of occurrences of j in d.  The maximal strong index is called the
    Durfee index.

    This particular rearrangement comes from the proof of Theorem 3 in [2]_.

    The ZZ condition says that for the sequence d if

    .. math::
        |d| >= \\frac{(\\max(d) + \\min(d) + 1)^2}{4*\\min(d)}

    then d is graphical.  This was shown in Theorem 6 in [2]_.

    References
    ----------
    .. [1] A. Tripathi and S. Vijay. "A note on a theorem of Erdős & Gallai",
       Discrete Mathematics, 265, pp. 417-420 (2003).
    .. [2] I.E. Zverovich and V.E. Zverovich. "Contributions to the theory
       of graphic sequences", Discrete Mathematics, 105, pp. 292-303 (1992).
    .. [EG1960] Erdős and Gallai, Mat. Lapok 11 264, 1960.
    """
    try:
        dmax, dmin, dsum, n, num_degs = _basic_graphical_tests(deg_sequence)
    except nx.NetworkXUnfeasible:
        return False
    if n == 0 or 4 * dmin * n >= (dmax + dmin + 1) * (dmax + dmin + 1):
        return True
    k, sum_deg, sum_nj, sum_jnj = (0, 0, 0, 0)
    for dk in range(dmax, dmin - 1, -1):
        if dk < k + 1:
            return True
        if num_degs[dk] > 0:
            run_size = num_degs[dk]
            if dk < k + run_size:
                run_size = dk - k
            sum_deg += run_size * dk
            for v in range(run_size):
                sum_nj += num_degs[k + v]
                sum_jnj += (k + v) * num_degs[k + v]
            k += run_size
            if sum_deg > k * (n - 1) - k * sum_nj + sum_jnj:
                return False
    return True
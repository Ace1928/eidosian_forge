import heapq
import networkx as nx
@nx._dispatch(graphs=None)
def is_digraphical(in_sequence, out_sequence):
    """Returns True if some directed graph can realize the in- and out-degree
    sequences.

    Parameters
    ----------
    in_sequence : list or iterable container
        A sequence of integer node in-degrees

    out_sequence : list or iterable container
        A sequence of integer node out-degrees

    Returns
    -------
    valid : bool
      True if in and out-sequences are digraphic False if not.

    Examples
    --------
    >>> G = nx.DiGraph([(1, 2), (1, 3), (2, 3), (3, 4), (4, 2), (5, 1), (5, 4)])
    >>> in_seq = (d for n, d in G.in_degree())
    >>> out_seq = (d for n, d in G.out_degree())
    >>> nx.is_digraphical(in_seq, out_seq)
    True

    To test a non-digraphical scenario:
    >>> in_seq_list = [d for n, d in G.in_degree()]
    >>> in_seq_list[-1] += 1
    >>> nx.is_digraphical(in_seq_list, out_seq)
    False

    Notes
    -----
    This algorithm is from Kleitman and Wang [1]_.
    The worst case runtime is $O(s \\times \\log n)$ where $s$ and $n$ are the
    sum and length of the sequences respectively.

    References
    ----------
    .. [1] D.J. Kleitman and D.L. Wang
       Algorithms for Constructing Graphs and Digraphs with Given Valences
       and Factors, Discrete Mathematics, 6(1), pp. 79-88 (1973)
    """
    try:
        in_deg_sequence = nx.utils.make_list_of_ints(in_sequence)
        out_deg_sequence = nx.utils.make_list_of_ints(out_sequence)
    except nx.NetworkXError:
        return False
    sumin, sumout, nin, nout = (0, 0, len(in_deg_sequence), len(out_deg_sequence))
    maxn = max(nin, nout)
    maxin = 0
    if maxn == 0:
        return True
    stubheap, zeroheap = ([], [])
    for n in range(maxn):
        in_deg, out_deg = (0, 0)
        if n < nout:
            out_deg = out_deg_sequence[n]
        if n < nin:
            in_deg = in_deg_sequence[n]
        if in_deg < 0 or out_deg < 0:
            return False
        sumin, sumout, maxin = (sumin + in_deg, sumout + out_deg, max(maxin, in_deg))
        if in_deg > 0:
            stubheap.append((-1 * out_deg, -1 * in_deg))
        elif out_deg > 0:
            zeroheap.append(-1 * out_deg)
    if sumin != sumout:
        return False
    heapq.heapify(stubheap)
    heapq.heapify(zeroheap)
    modstubs = [(0, 0)] * (maxin + 1)
    while stubheap:
        freeout, freein = heapq.heappop(stubheap)
        freein *= -1
        if freein > len(stubheap) + len(zeroheap):
            return False
        mslen = 0
        for i in range(freein):
            if zeroheap and (not stubheap or stubheap[0][0] > zeroheap[0]):
                stubout = heapq.heappop(zeroheap)
                stubin = 0
            else:
                stubout, stubin = heapq.heappop(stubheap)
            if stubout == 0:
                return False
            if stubout + 1 < 0 or stubin < 0:
                modstubs[mslen] = (stubout + 1, stubin)
                mslen += 1
        for i in range(mslen):
            stub = modstubs[i]
            if stub[1] < 0:
                heapq.heappush(stubheap, stub)
            else:
                heapq.heappush(zeroheap, stub[0])
        if freeout < 0:
            heapq.heappush(zeroheap, freeout)
    return True
import networkx as nx
from networkx.exception import NetworkXError
from networkx.readwrite.graph6 import data_to_n, n_to_data
from networkx.utils import not_implemented_for, open_file
@nx._dispatch(graphs=None)
def from_sparse6_bytes(string):
    """Read an undirected graph in sparse6 format from string.

    Parameters
    ----------
    string : string
       Data in sparse6 format

    Returns
    -------
    G : Graph

    Raises
    ------
    NetworkXError
        If the string is unable to be parsed in sparse6 format

    Examples
    --------
    >>> G = nx.from_sparse6_bytes(b":A_")
    >>> sorted(G.edges())
    [(0, 1), (0, 1), (0, 1)]

    See Also
    --------
    read_sparse6, write_sparse6

    References
    ----------
    .. [1] Sparse6 specification
           <https://users.cecs.anu.edu.au/~bdm/data/formats.html>

    """
    if string.startswith(b'>>sparse6<<'):
        string = string[11:]
    if not string.startswith(b':'):
        raise NetworkXError('Expected leading colon in sparse6')
    chars = [c - 63 for c in string[1:]]
    n, data = data_to_n(chars)
    k = 1
    while 1 << k < n:
        k += 1

    def parseData():
        """Returns stream of pairs b[i], x[i] for sparse6 format."""
        chunks = iter(data)
        d = None
        dLen = 0
        while 1:
            if dLen < 1:
                try:
                    d = next(chunks)
                except StopIteration:
                    return
                dLen = 6
            dLen -= 1
            b = d >> dLen & 1
            x = d & (1 << dLen) - 1
            xLen = dLen
            while xLen < k:
                try:
                    d = next(chunks)
                except StopIteration:
                    return
                dLen = 6
                x = (x << 6) + d
                xLen += 6
            x = x >> xLen - k
            dLen = xLen - k
            yield (b, x)
    v = 0
    G = nx.MultiGraph()
    G.add_nodes_from(range(n))
    multigraph = False
    for b, x in parseData():
        if b == 1:
            v += 1
        if x >= n or v >= n:
            break
        elif x > v:
            v = x
        else:
            if G.has_edge(x, v):
                multigraph = True
            G.add_edge(x, v)
    if not multigraph:
        G = nx.Graph(G)
    return G
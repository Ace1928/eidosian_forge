import networkx as nx
from networkx.exception import NetworkXError
from networkx.readwrite.graph6 import data_to_n, n_to_data
from networkx.utils import not_implemented_for, open_file
@open_file(0, mode='rb')
@nx._dispatch(graphs=None)
def read_sparse6(path):
    """Read an undirected graph in sparse6 format from path.

    Parameters
    ----------
    path : file or string
       File or filename to write.

    Returns
    -------
    G : Graph/Multigraph or list of Graphs/MultiGraphs
       If the file contains multiple lines then a list of graphs is returned

    Raises
    ------
    NetworkXError
        If the string is unable to be parsed in sparse6 format

    Examples
    --------
    You can read a sparse6 file by giving the path to the file::

        >>> import tempfile
        >>> with tempfile.NamedTemporaryFile(delete=False) as f:
        ...     _ = f.write(b">>sparse6<<:An\\n")
        ...     _ = f.seek(0)
        ...     G = nx.read_sparse6(f.name)
        >>> list(G.edges())
        [(0, 1)]

    You can also read a sparse6 file by giving an open file-like object::

        >>> import tempfile
        >>> with tempfile.NamedTemporaryFile() as f:
        ...     _ = f.write(b">>sparse6<<:An\\n")
        ...     _ = f.seek(0)
        ...     G = nx.read_sparse6(f)
        >>> list(G.edges())
        [(0, 1)]

    See Also
    --------
    read_sparse6, from_sparse6_bytes

    References
    ----------
    .. [1] Sparse6 specification
           <https://users.cecs.anu.edu.au/~bdm/data/formats.html>

    """
    glist = []
    for line in path:
        line = line.strip()
        if not len(line):
            continue
        glist.append(from_sparse6_bytes(line))
    if len(glist) == 1:
        return glist[0]
    else:
        return glist
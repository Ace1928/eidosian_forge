import networkx as nx
from networkx.exception import NetworkXError
from networkx.utils import open_file
@open_file(0, mode='rb')
@nx._dispatch(graphs=None)
def read_leda(path, encoding='UTF-8'):
    """Read graph in LEDA format from path.

    Parameters
    ----------
    path : file or string
       File or filename to read.  Filenames ending in .gz or .bz2  will be
       uncompressed.

    Returns
    -------
    G : NetworkX graph

    Examples
    --------
    G=nx.read_leda('file.leda')

    References
    ----------
    .. [1] http://www.algorithmic-solutions.info/leda_guide/graphs/leda_native_graph_fileformat.html
    """
    lines = (line.decode(encoding) for line in path)
    G = parse_leda(lines)
    return G
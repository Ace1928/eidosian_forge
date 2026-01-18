import os
import tempfile
import networkx as nx
@nx._dispatch(name='agraph_read_dot', graphs=None)
def read_dot(path):
    """Returns a NetworkX graph from a dot file on path.

    Parameters
    ----------
    path : file or string
       File name or file handle to read.
    """
    try:
        import pygraphviz
    except ImportError as err:
        raise ImportError('read_dot() requires pygraphviz http://pygraphviz.github.io/') from err
    A = pygraphviz.AGraph(file=path)
    gr = from_agraph(A)
    A.clear()
    return gr
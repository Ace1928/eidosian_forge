from copy import deepcopy
from functools import cached_property
import networkx as nx
from networkx import convert
from networkx.classes.coreviews import AdjacencyView
from networkx.classes.reportviews import DegreeView, EdgeView, NodeView
from networkx.exception import NetworkXError
def nbunch_iter(self, nbunch=None):
    """Returns an iterator over nodes contained in nbunch that are
        also in the graph.

        The nodes in nbunch are checked for membership in the graph
        and if not are silently ignored.

        Parameters
        ----------
        nbunch : single node, container, or all nodes (default= all nodes)
            The view will only report edges incident to these nodes.

        Returns
        -------
        niter : iterator
            An iterator over nodes in nbunch that are also in the graph.
            If nbunch is None, iterate over all nodes in the graph.

        Raises
        ------
        NetworkXError
            If nbunch is not a node or sequence of nodes.
            If a node in nbunch is not hashable.

        See Also
        --------
        Graph.__iter__

        Notes
        -----
        When nbunch is an iterator, the returned iterator yields values
        directly from nbunch, becoming exhausted when nbunch is exhausted.

        To test whether nbunch is a single node, one can use
        "if nbunch in self:", even after processing with this routine.

        If nbunch is not a node or a (possibly empty) sequence/iterator
        or None, a :exc:`NetworkXError` is raised.  Also, if any object in
        nbunch is not hashable, a :exc:`NetworkXError` is raised.
        """
    if nbunch is None:
        bunch = iter(self._adj)
    elif nbunch in self:
        bunch = iter([nbunch])
    else:

        def bunch_iter(nlist, adj):
            try:
                for n in nlist:
                    if n in adj:
                        yield n
            except TypeError as err:
                exc, message = (err, err.args[0])
                if 'iter' in message:
                    exc = NetworkXError('nbunch is not a node or a sequence of nodes.')
                if 'hashable' in message:
                    exc = NetworkXError(f'Node {n} in sequence nbunch is not a valid node.')
                raise exc
        bunch = bunch_iter(nbunch, self._adj)
    return bunch
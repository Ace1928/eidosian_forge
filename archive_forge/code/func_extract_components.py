from collections import Counter
import numpy as np
from scipy import sparse
from pygsp import utils
from . import fourier, difference  # prevent circular import in Python < 3.5
def extract_components(self):
    """Split the graph into connected components.

        See :func:`is_connected` for the method used to determine
        connectedness.

        Returns
        -------
        graphs : list
            A list of graph structures. Each having its own node list and
            weight matrix. If the graph is directed, add into the info
            parameter the information about the source nodes and the sink
            nodes.

        Examples
        --------
        >>> from scipy import sparse
        >>> W = sparse.rand(10, 10, 0.2)
        >>> W = utils.symmetrize(W)
        >>> G = graphs.Graph(W=W)
        >>> components = G.extract_components()
        >>> has_sinks = 'sink' in components[0].info
        >>> sinks_0 = components[0].info['sink'] if has_sinks else []

        """
    if self.A.shape[0] != self.A.shape[1]:
        self.logger.error('Inconsistent shape to extract components. Square matrix required.')
        return None
    if self.is_directed():
        raise NotImplementedError('Directed graphs not supported yet.')
    graphs = []
    visited = np.zeros(self.A.shape[0], dtype=bool)
    while not visited.all():
        stack = set(np.nonzero(~visited)[0])
        comp = []
        while len(stack):
            v = stack.pop()
            if not visited[v]:
                comp.append(v)
                visited[v] = True
                stack.update(set([idx for idx in self.A[v, :].nonzero()[1] if not visited[idx]]))
        comp = sorted(comp)
        self.logger.info('Constructing subgraph for component of size {}.'.format(len(comp)))
        G = self.subgraph(comp)
        G.info = {'orig_idx': comp}
        graphs.append(G)
    return graphs